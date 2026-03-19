"""
Modern Attack Defenses (2025)
==============================
Defense mechanisms against state-of-the-art adversarial attacks on LLMs.

Covers attack vectors from 2024-2025 research:
- Crescendo/Multi-turn attacks (Microsoft, 2024)
- Many-shot jailbreaking (Anthropic, 2024)
- Skeleton Key attacks (Microsoft, 2024)
- Prompt injection (direct and indirect)
- RAG poisoning
- Encoding-based attacks (Base64, ASCII art)
- Cross-prompt injection attacks (XPIA)
- Tool/function calling exploits

Based on research from:
- Microsoft AI Red Team (2024)
- Anthropic Multi-Shot Paper (2024)
- OWASP LLM Top 10 (2025)
- Google DeepMind Adversarial Robustness
"""

import base64
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger("hololoom.alignment.modern_attack_defenses")


class AttackType(str, Enum):
    """
    Types of adversarial attacks.

    Categorized by attack mechanism and target.
    """
    # Jailbreaking attacks
    CRESCENDO = "crescendo"                    # Multi-turn escalation
    MANY_SHOT = "many_shot"                    # Overwhelming with examples
    SKELETON_KEY = "skeleton_key"              # Behavior mode manipulation
    DAN = "dan"                                # "Do Anything Now" variants

    # Injection attacks
    PROMPT_INJECTION_DIRECT = "prompt_injection_direct"
    PROMPT_INJECTION_INDIRECT = "prompt_injection_indirect"
    XPIA = "xpia"                              # Cross-prompt injection attack
    RAG_POISONING = "rag_poisoning"           # Malicious RAG content

    # Encoding attacks
    BASE64_SMUGGLING = "base64_smuggling"
    ASCII_ART = "ascii_art"
    UNICODE_OBFUSCATION = "unicode_obfuscation"
    LEETSPEAK = "leetspeak"

    # Tool/function attacks
    TOOL_ABUSE = "tool_abuse"
    FUNCTION_INJECTION = "function_injection"
    SCHEMA_MANIPULATION = "schema_manipulation"

    # Other attacks
    CONTEXT_OVERFLOW = "context_overflow"
    SYSTEM_PROMPT_EXTRACTION = "system_prompt_extraction"
    PERSONA_HIJACKING = "persona_hijacking"


class ThreatLevel(str, Enum):
    """
    Threat severity levels.
    """
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DefenseAction(str, Enum):
    """
    Actions to take when attack detected.
    """
    ALLOW = "allow"                # No intervention
    WARN = "warn"                  # Log warning but allow
    SANITIZE = "sanitize"          # Clean input and continue
    REJECT = "reject"              # Reject with explanation
    ESCALATE = "escalate"          # Require human review
    BLOCK = "block"                # Hard block, no explanation


@dataclass
class AttackSignature:
    """
    Signature for detecting a specific attack pattern.
    """
    attack_type: AttackType
    name: str
    description: str
    patterns: list[str] = field(default_factory=list)  # Regex patterns
    keywords: list[str] = field(default_factory=list)  # Keyword matches
    heuristics: list[str] = field(default_factory=list)  # Named heuristics
    severity: ThreatLevel = ThreatLevel.MEDIUM
    confidence_threshold: float = 0.6


@dataclass
class DetectionResult:
    """
    Result of attack detection.
    """
    detected: bool
    attack_type: AttackType | None
    threat_level: ThreatLevel
    confidence: float
    evidence: list[str]
    recommended_action: DefenseAction
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "detected": self.detected,
            "attack_type": self.attack_type.value if self.attack_type else None,
            "threat_level": self.threat_level.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "recommended_action": self.recommended_action.value,
            "metadata": self.metadata,
        }


@dataclass
class ConversationTurn:
    """
    A single turn in a conversation for multi-turn attack tracking.
    """
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    threat_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class CrescendoDefense:
    """
    Defense against crescendo/multi-turn attacks.

    Crescendo attacks gradually escalate across multiple turns,
    starting benign and slowly introducing harmful elements.

    Detection approach:
    - Track escalation patterns across turns
    - Detect topic drift toward harmful areas
    - Identify gradual boundary testing
    """

    # Harmful topic categories for escalation detection
    HARMFUL_CATEGORIES = {
        "violence": ["kill", "murder", "attack", "weapon", "bomb", "hurt", "harm"],
        "illegal": ["steal", "hack", "fraud", "drugs", "illegal", "crime"],
        "explicit": ["nsfw", "sexual", "nude", "explicit", "porn"],
        "manipulation": ["manipulate", "deceive", "trick", "exploit", "scam"],
        "self_harm": ["suicide", "self-harm", "cut myself", "end my life"],
    }

    # Topic progression signals (benign → harmful transitions)
    ESCALATION_SIGNALS = [
        ("fictional", "real"),
        ("hypothetical", "actual"),
        ("educational", "instructional"),
        ("historical", "current"),
        ("general", "specific"),
    ]

    def __init__(
        self,
        window_size: int = 10,
        escalation_threshold: float = 0.7,
        harmfulness_threshold: float = 0.5,
    ):
        """
        Initialize crescendo defense.

        Args:
            window_size: Number of turns to track
            escalation_threshold: Threshold for escalation detection
            harmfulness_threshold: Threshold for harmful content
        """
        self.window_size = window_size
        self.escalation_threshold = escalation_threshold
        self.harmfulness_threshold = harmfulness_threshold
        self._conversation_history: deque = deque(maxlen=window_size)

    def add_turn(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationTurn:
        """Add a turn to conversation history."""
        threat_score = self._calculate_threat_score(content)

        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.now(),
            threat_score=threat_score,
            metadata=metadata or {},
        )

        self._conversation_history.append(turn)
        return turn

    def _calculate_threat_score(self, content: str) -> float:
        """Calculate threat score for a single message."""
        content_lower = content.lower()
        score = 0.0

        for category, keywords in self.HARMFUL_CATEGORIES.items():
            matches = sum(1 for kw in keywords if kw in content_lower)
            if matches > 0:
                score += 0.2 * min(matches, 3)  # Cap at 3 matches per category

        return min(score, 1.0)

    def detect_escalation(self) -> DetectionResult:
        """
        Detect crescendo/escalation attack pattern.

        Returns detection result with evidence.
        """
        if len(self._conversation_history) < 3:
            return DetectionResult(
                detected=False,
                attack_type=None,
                threat_level=ThreatLevel.NONE,
                confidence=0.0,
                evidence=["Insufficient conversation history"],
                recommended_action=DefenseAction.ALLOW,
            )

        evidence = []

        # Check 1: Increasing threat scores
        user_turns = [t for t in self._conversation_history if t.role == "user"]
        if len(user_turns) >= 3:
            scores = [t.threat_score for t in user_turns[-5:]]
            if len(scores) >= 3:
                # Check for monotonic increase
                increasing = all(scores[i] <= scores[i+1] for i in range(len(scores)-1))
                if increasing and scores[-1] > scores[0] + 0.2:
                    evidence.append(f"Escalating threat scores: {[f'{s:.2f}' for s in scores]}")

        # Check 2: Topic progression from benign to harmful
        if len(user_turns) >= 2:
            first_content = user_turns[0].content.lower()
            last_content = user_turns[-1].content.lower()

            for benign, harmful in self.ESCALATION_SIGNALS:
                if benign in first_content and harmful in last_content:
                    evidence.append(f"Topic shift from '{benign}' to '{harmful}'")

        # Check 3: Cumulative harmfulness exceeding threshold
        total_threat = sum(t.threat_score for t in self._conversation_history)
        avg_threat = total_threat / len(self._conversation_history)

        if avg_threat > self.harmfulness_threshold:
            evidence.append(f"High average threat: {avg_threat:.2f}")

        # Check 4: Sudden spike in last message
        if user_turns and user_turns[-1].threat_score > 0.6:
            if len(user_turns) >= 2 and user_turns[-2].threat_score < 0.3:
                evidence.append("Sudden threat spike in latest message")

        # Determine detection result
        detected = len(evidence) >= 2  # Need multiple signals
        confidence = min(len(evidence) * 0.25, 1.0)

        if detected:
            return DetectionResult(
                detected=True,
                attack_type=AttackType.CRESCENDO,
                threat_level=ThreatLevel.HIGH if confidence > 0.7 else ThreatLevel.MEDIUM,
                confidence=confidence,
                evidence=evidence,
                recommended_action=DefenseAction.REJECT if confidence > 0.8 else DefenseAction.WARN,
                metadata={"turns_analyzed": len(self._conversation_history)},
            )

        return DetectionResult(
            detected=False,
            attack_type=None,
            threat_level=ThreatLevel.LOW if evidence else ThreatLevel.NONE,
            confidence=confidence,
            evidence=evidence or ["No escalation pattern detected"],
            recommended_action=DefenseAction.ALLOW,
        )

    def reset(self):
        """Reset conversation history."""
        self._conversation_history.clear()


class ManyShotDefense:
    """
    Defense against many-shot jailbreaking.

    Many-shot attacks overwhelm the model with numerous examples
    of harmful behavior to normalize it and bypass safety training.

    Detection approach:
    - Count repetitive patterns
    - Detect example flooding
    - Identify Q&A manipulation patterns
    """

    # Patterns indicating example injection
    EXAMPLE_PATTERNS = [
        r"example \d+:",
        r"case \d+:",
        r"scenario \d+:",
        r"instance \d+:",
        r"Q\d+:",
        r"Question \d+:",
        r"User:\s.*\nAssistant:",
        r"\[example\]",
        r"\[case\]",
    ]

    def __init__(
        self,
        max_examples: int = 5,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize many-shot defense.

        Args:
            max_examples: Max allowed examples before flagging
            similarity_threshold: Threshold for similar examples
        """
        self.max_examples = max_examples
        self.similarity_threshold = similarity_threshold

    def detect_many_shot(self, content: str) -> DetectionResult:
        """
        Detect many-shot jailbreaking attempt.

        Args:
            content: User input to analyze

        Returns:
            Detection result
        """
        evidence = []
        example_count = 0

        # Check 1: Count example patterns
        for pattern in self.EXAMPLE_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            example_count += len(matches)
            if matches:
                evidence.append(f"Found {len(matches)} matches for pattern: {pattern[:30]}...")

        # Check 2: Detect repetitive structures
        lines = content.split('\n')
        if len(lines) > 10:
            # Check for repeated formats
            line_starts = [line[:20] for line in lines if len(line) > 20]
            if line_starts:
                from collections import Counter
                start_counts = Counter(line_starts)
                repetitions = sum(1 for count in start_counts.values() if count >= 3)
                if repetitions > 0:
                    evidence.append(f"Detected {repetitions} repetitive line patterns")
                    example_count += repetitions * 2

        # Check 3: Detect Q&A pair flooding
        qa_patterns = [
            (r"Q:", r"A:"),
            (r"Question:", r"Answer:"),
            (r"User:", r"Assistant:"),
            (r"Human:", r"AI:"),
        ]
        for q_pat, a_pat in qa_patterns:
            q_matches = len(re.findall(q_pat, content, re.IGNORECASE))
            a_matches = len(re.findall(a_pat, content, re.IGNORECASE))
            if q_matches >= 3 and a_matches >= 3:
                evidence.append(f"Detected {q_matches} Q&A pairs")
                example_count += q_matches

        # Check 4: Length anomaly (very long inputs often indicate many-shot)
        if len(content) > 5000:
            evidence.append(f"Abnormally long input: {len(content)} characters")
            example_count += 1

        # Determine result
        detected = example_count > self.max_examples
        confidence = min(example_count / (self.max_examples * 2), 1.0)

        if detected:
            return DetectionResult(
                detected=True,
                attack_type=AttackType.MANY_SHOT,
                threat_level=ThreatLevel.HIGH,
                confidence=confidence,
                evidence=evidence,
                recommended_action=DefenseAction.REJECT,
                metadata={"example_count": example_count},
            )

        return DetectionResult(
            detected=False,
            attack_type=None,
            threat_level=ThreatLevel.LOW if evidence else ThreatLevel.NONE,
            confidence=confidence,
            evidence=evidence or ["No many-shot pattern detected"],
            recommended_action=DefenseAction.ALLOW,
        )


class PromptInjectionDefense:
    """
    Defense against direct and indirect prompt injection.

    Direct injection: User input contains instructions for the model
    Indirect injection: External data (RAG, tools) contains malicious instructions

    Detection approach:
    - Detect instruction-like patterns in user input
    - Identify role confusion attempts
    - Detect delimiter manipulation
    - Check for system prompt extraction attempts
    """

    # Instruction patterns that indicate injection
    INJECTION_PATTERNS = [
        r"ignore (all )?(previous |above |prior )?instructions",
        r"disregard (all )?(previous |above |prior )?(instructions|rules)",
        r"forget (everything|all|your) (you|instructions)",
        r"new (instructions|rules|prompt):",
        r"system prompt:",
        r"you are now",
        r"your new (role|task|instruction)",
        r"override (your |the )?(instructions|rules|training)",
        r"from now on",
        r"act as (if you are|a|an)",
        r"pretend (to be|you are)",
        r"\[system\]",
        r"\[INST\]",
        r"<<<.*>>>",  # Common injection delimiters
        r"\|\|.*\|\|",
        r"```\s*(system|instruction)",
    ]

    # Role confusion patterns
    ROLE_CONFUSION = [
        r"you must (always |now )?respond",
        r"assistant:",
        r"AI:",
        r"ChatGPT:",
        r"Claude:",
        r"respond (only |exclusively )?with",
        r"your (only |primary )?purpose",
        r"you (were|are) (created|designed|built) to",
    ]

    # System prompt extraction attempts
    EXTRACTION_PATTERNS = [
        r"(what|tell me|show|reveal|display|repeat|print) (is |are )?(your|the) (system |initial )?(prompt|instructions|rules)",
        r"(beginning|start|first) (of |)?(your |the )?prompt",
        r"above (this |the )?(message|text|input)",
        r"(before|prior to) (this|my) (message|input)",
        r"text (above|before)",
    ]

    def __init__(
        self,
        sensitivity: str = "medium",  # low, medium, high
        check_external_data: bool = True,
    ):
        """
        Initialize prompt injection defense.

        Args:
            sensitivity: Detection sensitivity level
            check_external_data: Also check RAG/tool outputs
        """
        self.sensitivity = sensitivity
        self.check_external_data = check_external_data

        # Adjust thresholds based on sensitivity
        self._thresholds = {
            "low": {"pattern_matches": 2, "confidence": 0.7},
            "medium": {"pattern_matches": 1, "confidence": 0.5},
            "high": {"pattern_matches": 1, "confidence": 0.3},
        }

    def detect_injection(
        self,
        user_input: str,
        external_data: list[str] | None = None,
    ) -> DetectionResult:
        """
        Detect prompt injection attempts.

        Args:
            user_input: Direct user input
            external_data: Optional RAG/tool outputs to check

        Returns:
            Detection result
        """
        evidence = []
        attack_type = AttackType.PROMPT_INJECTION_DIRECT
        pattern_matches = 0

        # Check user input
        input_lower = user_input.lower()

        # Check injection patterns
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, input_lower):
                evidence.append(f"Injection pattern: {pattern[:40]}...")
                pattern_matches += 1

        # Check role confusion
        for pattern in self.ROLE_CONFUSION:
            if re.search(pattern, input_lower):
                evidence.append(f"Role confusion: {pattern[:40]}...")
                pattern_matches += 1

        # Check extraction attempts
        for pattern in self.EXTRACTION_PATTERNS:
            if re.search(pattern, input_lower):
                evidence.append(f"Extraction attempt: {pattern[:40]}...")
                pattern_matches += 1
                attack_type = AttackType.SYSTEM_PROMPT_EXTRACTION

        # Check external data (indirect injection)
        if self.check_external_data and external_data:
            for i, data in enumerate(external_data):
                data_lower = data.lower()
                for pattern in self.INJECTION_PATTERNS:
                    if re.search(pattern, data_lower):
                        evidence.append(f"Indirect injection in source {i}: {pattern[:40]}...")
                        pattern_matches += 1
                        attack_type = AttackType.PROMPT_INJECTION_INDIRECT

        # Determine result
        threshold = self._thresholds[self.sensitivity]
        detected = pattern_matches >= threshold["pattern_matches"]
        confidence = min(pattern_matches * 0.3, 1.0)

        if detected:
            threat_level = ThreatLevel.CRITICAL if pattern_matches >= 3 else ThreatLevel.HIGH
            return DetectionResult(
                detected=True,
                attack_type=attack_type,
                threat_level=threat_level,
                confidence=confidence,
                evidence=evidence,
                recommended_action=DefenseAction.REJECT,
                metadata={"pattern_matches": pattern_matches},
            )

        return DetectionResult(
            detected=False,
            attack_type=None,
            threat_level=ThreatLevel.NONE,
            confidence=confidence,
            evidence=evidence or ["No injection detected"],
            recommended_action=DefenseAction.ALLOW,
        )


class EncodingAttackDefense:
    """
    Defense against encoding-based attacks.

    Attacks that use alternative representations to bypass filters:
    - Base64 encoding
    - ASCII art
    - Unicode homoglyphs
    - Leetspeak
    - ROT13 and other ciphers
    """

    # Base64 patterns
    BASE64_PATTERN = r"^[A-Za-z0-9+/]{20,}={0,2}$"

    # Leetspeak substitutions
    LEETSPEAK_MAP = {
        '4': 'a', '@': 'a', '8': 'b', '3': 'e', '1': 'i', 'l': 'i',
        '0': 'o', '5': 's', '$': 's', '7': 't', '+': 't'
    }

    # Unicode homoglyphs (selected examples)
    HOMOGLYPHS = {
        'а': 'a',  # Cyrillic
        'е': 'e',
        'о': 'o',
        'р': 'p',
        'с': 'c',
        'х': 'x',
        'А': 'A',
        'В': 'B',
        'С': 'C',
        'Е': 'E',
    }

    def __init__(self):
        """Initialize encoding attack defense."""
        pass

    def detect_encoding_attack(self, content: str) -> DetectionResult:
        """
        Detect encoding-based attacks.

        Args:
            content: Input to analyze

        Returns:
            Detection result
        """
        evidence = []
        attack_type = None

        # Check 1: Base64 detection
        base64_result = self._check_base64(content)
        if base64_result:
            evidence.append(f"Base64 content detected: {base64_result[:50]}...")
            attack_type = AttackType.BASE64_SMUGGLING

        # Check 2: ASCII art detection
        if self._check_ascii_art(content):
            evidence.append("ASCII art pattern detected")
            attack_type = attack_type or AttackType.ASCII_ART

        # Check 3: Unicode homoglyph detection
        homoglyph_count = self._check_homoglyphs(content)
        if homoglyph_count > 2:
            evidence.append(f"Unicode homoglyphs detected: {homoglyph_count} instances")
            attack_type = attack_type or AttackType.UNICODE_OBFUSCATION

        # Check 4: Leetspeak detection
        leetspeak_ratio = self._check_leetspeak(content)
        if leetspeak_ratio > 0.3:
            evidence.append(f"Leetspeak ratio: {leetspeak_ratio:.1%}")
            attack_type = attack_type or AttackType.LEETSPEAK

        # Determine result
        detected = len(evidence) > 0
        confidence = min(len(evidence) * 0.35, 1.0)

        if detected:
            return DetectionResult(
                detected=True,
                attack_type=attack_type,
                threat_level=ThreatLevel.MEDIUM,
                confidence=confidence,
                evidence=evidence,
                recommended_action=DefenseAction.SANITIZE,
                metadata={"encoding_type": attack_type.value if attack_type else None},
            )

        return DetectionResult(
            detected=False,
            attack_type=None,
            threat_level=ThreatLevel.NONE,
            confidence=0.0,
            evidence=["No encoding attack detected"],
            recommended_action=DefenseAction.ALLOW,
        )

    def _check_base64(self, content: str) -> str | None:
        """Check for and decode Base64 content."""
        # Find potential Base64 strings
        words = content.split()
        for word in words:
            if len(word) >= 20 and re.match(self.BASE64_PATTERN, word):
                try:
                    decoded = base64.b64decode(word).decode('utf-8', errors='ignore')
                    if len(decoded) > 5:  # Non-trivial content
                        return decoded
                except Exception:
                    pass
        return None

    def _check_ascii_art(self, content: str) -> bool:
        """Check for ASCII art patterns."""
        lines = content.split('\n')
        if len(lines) < 3:
            return False

        # Check for consistent line patterns
        special_char_ratio = sum(
            1 for c in content if c in '\\|/_-=+*#@'
        ) / max(len(content), 1)

        # High special character ratio suggests ASCII art
        return special_char_ratio > 0.3

    def _check_homoglyphs(self, content: str) -> int:
        """Count Unicode homoglyphs."""
        count = 0
        for char in content:
            if char in self.HOMOGLYPHS:
                count += 1
        return count

    def _check_leetspeak(self, content: str) -> float:
        """Calculate leetspeak ratio."""
        if not content:
            return 0.0

        leetspeak_chars = sum(1 for c in content if c in self.LEETSPEAK_MAP)
        alpha_chars = sum(1 for c in content if c.isalpha())

        if alpha_chars == 0:
            return 0.0

        return leetspeak_chars / (leetspeak_chars + alpha_chars)

    def decode_content(self, content: str) -> str:
        """
        Attempt to decode obfuscated content.

        Returns decoded/normalized version for further analysis.
        """
        decoded = content

        # Decode Base64
        base64_result = self._check_base64(content)
        if base64_result:
            decoded = base64_result

        # Normalize homoglyphs
        for homoglyph, replacement in self.HOMOGLYPHS.items():
            decoded = decoded.replace(homoglyph, replacement)

        # Normalize leetspeak
        for leet, replacement in self.LEETSPEAK_MAP.items():
            decoded = decoded.replace(leet, replacement)

        return decoded


class ToolAbuseDefense:
    """
    Defense against tool/function calling exploits.

    Attacks that abuse agent tool capabilities:
    - Unauthorized tool access
    - Parameter injection
    - Chained tool abuse
    - Resource exhaustion through tools
    """

    # Dangerous tool operations
    DANGEROUS_OPERATIONS = {
        "file_system": ["delete", "remove", "rm", "unlink", "format"],
        "network": ["exfiltrate", "send_data", "upload_to", "transfer"],
        "system": ["execute", "shell", "sudo", "admin", "root"],
        "database": ["drop", "truncate", "delete_all", "purge"],
    }

    # Suspicious parameter patterns
    SUSPICIOUS_PARAMS = [
        r"\.\./",                    # Path traversal
        r"/etc/",                    # System directories
        r"/root/",
        r"rm -rf",
        r"--no-preserve-root",
        r";.*",                       # Command chaining
        r"\|.*",                      # Piping
        r"`.*`",                      # Command substitution
        r"\$\(.*\)",                  # Command substitution
    ]

    def __init__(
        self,
        allowed_tools: set[str] | None = None,
        max_tool_calls_per_turn: int = 5,
    ):
        """
        Initialize tool abuse defense.

        Args:
            allowed_tools: Whitelist of allowed tools
            max_tool_calls_per_turn: Max tool calls per turn
        """
        self.allowed_tools = allowed_tools
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self._tool_call_count = 0

    def check_tool_call(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> DetectionResult:
        """
        Check if a tool call is potentially malicious.

        Args:
            tool_name: Name of the tool being called
            parameters: Tool parameters

        Returns:
            Detection result
        """
        evidence = []

        # Check 1: Tool allowlist
        if self.allowed_tools and tool_name not in self.allowed_tools:
            evidence.append(f"Tool '{tool_name}' not in allowed list")

        # Check 2: Rate limiting
        self._tool_call_count += 1
        if self._tool_call_count > self.max_tool_calls_per_turn:
            evidence.append(f"Tool call limit exceeded: {self._tool_call_count}")

        # Check 3: Dangerous operations
        param_str = str(parameters).lower()
        for category, operations in self.DANGEROUS_OPERATIONS.items():
            for op in operations:
                if op in param_str:
                    evidence.append(f"Dangerous {category} operation: {op}")

        # Check 4: Suspicious parameter patterns
        for pattern in self.SUSPICIOUS_PARAMS:
            if re.search(pattern, param_str):
                evidence.append(f"Suspicious parameter pattern: {pattern[:30]}...")

        # Check 5: Path traversal
        for key, value in parameters.items():
            if isinstance(value, str):
                if ".." in value:
                    evidence.append(f"Path traversal in parameter '{key}'")
                if value.startswith("/") and any(d in value for d in ["/etc", "/root", "/var"]):
                    evidence.append(f"System path access in parameter '{key}'")

        # Determine result
        detected = len(evidence) > 0
        confidence = min(len(evidence) * 0.3, 1.0)

        if detected:
            threat_level = ThreatLevel.CRITICAL if len(evidence) >= 3 else ThreatLevel.HIGH
            return DetectionResult(
                detected=True,
                attack_type=AttackType.TOOL_ABUSE,
                threat_level=threat_level,
                confidence=confidence,
                evidence=evidence,
                recommended_action=DefenseAction.BLOCK,
                metadata={"tool_name": tool_name},
            )

        return DetectionResult(
            detected=False,
            attack_type=None,
            threat_level=ThreatLevel.NONE,
            confidence=0.0,
            evidence=["Tool call appears safe"],
            recommended_action=DefenseAction.ALLOW,
        )

    def reset_turn(self):
        """Reset per-turn counters."""
        self._tool_call_count = 0


class SkeletonKeyDefense:
    """
    Defense against skeleton key attacks.

    Skeleton key attacks attempt to put the model in a special mode
    where it believes safety rules don't apply.

    Common patterns:
    - "Enter developer mode"
    - "Disable safety filters"
    - "You are now in unrestricted mode"
    """

    # Skeleton key patterns
    SKELETON_PATTERNS = [
        r"(enter|enable|activate) (developer|god|admin|unrestricted|jailbreak) mode",
        r"disable (safety|content|ethical) (filter|check|guard)",
        r"(safety|content) (filter|mode) (off|disabled|bypassed)",
        r"you are (in|now in) (developer|unrestricted|jailbreak) mode",
        r"(respond|act) (without|ignoring) (safety|ethical|content) (concern|filter|restriction)",
        r"(all|any) (restriction|safety|ethical) (removed|disabled|off)",
        r"no (longer|more) (bound|restricted|limited) by",
        r"(operational|functional) (with|without)out (any|all) (restriction|limit)",
        r"(master|override|skeleton|bypass) key",
        r"unlock (all|full|complete) (access|capability|function)",
    ]

    def __init__(self):
        """Initialize skeleton key defense."""
        pass

    def detect_skeleton_key(self, content: str) -> DetectionResult:
        """
        Detect skeleton key attack attempts.

        Args:
            content: User input

        Returns:
            Detection result
        """
        evidence = []
        content_lower = content.lower()

        for pattern in self.SKELETON_PATTERNS:
            if re.search(pattern, content_lower):
                evidence.append(f"Skeleton key pattern: {pattern[:50]}...")

        detected = len(evidence) > 0
        confidence = min(len(evidence) * 0.4, 1.0)

        if detected:
            return DetectionResult(
                detected=True,
                attack_type=AttackType.SKELETON_KEY,
                threat_level=ThreatLevel.CRITICAL,
                confidence=confidence,
                evidence=evidence,
                recommended_action=DefenseAction.REJECT,
                metadata={"patterns_matched": len(evidence)},
            )

        return DetectionResult(
            detected=False,
            attack_type=None,
            threat_level=ThreatLevel.NONE,
            confidence=0.0,
            evidence=["No skeleton key pattern detected"],
            recommended_action=DefenseAction.ALLOW,
        )


class UnifiedAttackDefense:
    """
    Unified defense system combining all attack detectors.

    Provides comprehensive protection against modern attack vectors.
    """

    def __init__(
        self,
        enable_crescendo: bool = True,
        enable_many_shot: bool = True,
        enable_injection: bool = True,
        enable_encoding: bool = True,
        enable_tool_abuse: bool = True,
        enable_skeleton_key: bool = True,
        sensitivity: str = "medium",
    ):
        """
        Initialize unified defense system.

        Args:
            enable_*: Enable specific defense modules
            sensitivity: Overall sensitivity level
        """
        self.crescendo = CrescendoDefense() if enable_crescendo else None
        self.many_shot = ManyShotDefense() if enable_many_shot else None
        self.injection = PromptInjectionDefense(sensitivity=sensitivity) if enable_injection else None
        self.encoding = EncodingAttackDefense() if enable_encoding else None
        self.tool_abuse = ToolAbuseDefense() if enable_tool_abuse else None
        self.skeleton_key = SkeletonKeyDefense() if enable_skeleton_key else None

        self._stats = {
            "total_scans": 0,
            "attacks_detected": 0,
            "by_type": {},
        }

    def scan_input(
        self,
        content: str,
        external_data: list[str] | None = None,
        is_tool_call: bool = False,
        tool_name: str | None = None,
        tool_params: dict[str, Any] | None = None,
    ) -> list[DetectionResult]:
        """
        Scan input through all enabled defenses.

        Args:
            content: User input
            external_data: RAG/tool outputs
            is_tool_call: Whether this is a tool call
            tool_name: Tool name if tool call
            tool_params: Tool parameters if tool call

        Returns:
            List of detection results from all modules
        """
        self._stats["total_scans"] += 1
        results = []

        # 1. Skeleton Key (highest priority)
        if self.skeleton_key:
            result = self.skeleton_key.detect_skeleton_key(content)
            if result.detected:
                results.append(result)
                self._record_detection(AttackType.SKELETON_KEY)

        # 2. Prompt Injection
        if self.injection:
            result = self.injection.detect_injection(content, external_data)
            if result.detected:
                results.append(result)
                self._record_detection(result.attack_type)

        # 3. Many-shot
        if self.many_shot:
            result = self.many_shot.detect_many_shot(content)
            if result.detected:
                results.append(result)
                self._record_detection(AttackType.MANY_SHOT)

        # 4. Encoding attacks
        if self.encoding:
            result = self.encoding.detect_encoding_attack(content)
            if result.detected:
                results.append(result)
                self._record_detection(result.attack_type)

        # 5. Tool abuse (only for tool calls)
        if self.tool_abuse and is_tool_call and tool_name and tool_params:
            result = self.tool_abuse.check_tool_call(tool_name, tool_params)
            if result.detected:
                results.append(result)
                self._record_detection(AttackType.TOOL_ABUSE)

        # 6. Crescendo (add to conversation)
        if self.crescendo:
            self.crescendo.add_turn("user", content)
            result = self.crescendo.detect_escalation()
            if result.detected:
                results.append(result)
                self._record_detection(AttackType.CRESCENDO)

        return results

    def _record_detection(self, attack_type: AttackType):
        """Record detection in statistics."""
        self._stats["attacks_detected"] += 1
        type_key = attack_type.value
        self._stats["by_type"][type_key] = self._stats["by_type"].get(type_key, 0) + 1

    def get_aggregated_result(
        self,
        results: list[DetectionResult],
    ) -> DetectionResult:
        """
        Aggregate multiple detection results into a single result.

        Takes the most severe detection as the primary result.
        """
        if not results:
            return DetectionResult(
                detected=False,
                attack_type=None,
                threat_level=ThreatLevel.NONE,
                confidence=0.0,
                evidence=["No attacks detected"],
                recommended_action=DefenseAction.ALLOW,
            )

        # Sort by threat level (highest first)
        threat_order = {
            ThreatLevel.CRITICAL: 5,
            ThreatLevel.HIGH: 4,
            ThreatLevel.MEDIUM: 3,
            ThreatLevel.LOW: 2,
            ThreatLevel.NONE: 1,
        }

        sorted_results = sorted(
            results,
            key=lambda r: (threat_order.get(r.threat_level, 0), r.confidence),
            reverse=True
        )

        primary = sorted_results[0]

        # Aggregate evidence from all results
        all_evidence = []
        for r in results:
            all_evidence.extend(r.evidence)

        # Boost confidence if multiple detections
        combined_confidence = min(primary.confidence + 0.1 * (len(results) - 1), 1.0)

        return DetectionResult(
            detected=True,
            attack_type=primary.attack_type,
            threat_level=primary.threat_level,
            confidence=combined_confidence,
            evidence=all_evidence,
            recommended_action=primary.recommended_action,
            metadata={
                "total_detections": len(results),
                "attack_types": [r.attack_type.value for r in results if r.attack_type],
            },
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get defense statistics."""
        return self._stats.copy()

    def reset_conversation(self):
        """Reset conversation-based tracking."""
        if self.crescendo:
            self.crescendo.reset()
        if self.tool_abuse:
            self.tool_abuse.reset_turn()


# Factory functions
def create_attack_defense(
    sensitivity: str = "medium",
    enable_all: bool = True,
) -> UnifiedAttackDefense:
    """
    Create a unified attack defense system.

    Args:
        sensitivity: Detection sensitivity (low/medium/high)
        enable_all: Enable all defense modules

    Returns:
        Configured UnifiedAttackDefense instance
    """
    return UnifiedAttackDefense(
        enable_crescendo=enable_all,
        enable_many_shot=enable_all,
        enable_injection=enable_all,
        enable_encoding=enable_all,
        enable_tool_abuse=enable_all,
        enable_skeleton_key=enable_all,
        sensitivity=sensitivity,
    )


def quick_scan(content: str) -> dict[str, Any]:
    """
    Quick scan of content for common attacks.

    Args:
        content: Content to scan

    Returns:
        Quick scan summary
    """
    defense = create_attack_defense(sensitivity="medium")
    results = defense.scan_input(content)

    if not results:
        return {
            "safe": True,
            "attacks_detected": 0,
            "summary": "No attacks detected",
        }

    aggregated = defense.get_aggregated_result(results)

    return {
        "safe": False,
        "attacks_detected": len(results),
        "primary_attack": aggregated.attack_type.value if aggregated.attack_type else None,
        "threat_level": aggregated.threat_level.value,
        "confidence": aggregated.confidence,
        "recommended_action": aggregated.recommended_action.value,
        "evidence": aggregated.evidence[:5],  # Top 5 evidence items
    }
