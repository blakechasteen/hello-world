"""
REFUSE Loom - The Skeptic
==========================

Adversarial probing, breaking-grounded. Tries to find what's wrong.

Cognitive Style:
- Attack every claim
- Find hidden assumptions
- Probe for weaknesses
- Stress-test logic

Philosophy:
"What could go wrong? What are you not seeing?
The Skeptic assumes everything is wrong until proven otherwise."

Date: December 2025
Author: HoloLoom Architecture Team
"""

import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from hololoom.loom.base_loom import BaseLoom
from hololoom.loom.protocol import REFUSE
from hololoom.protocols.department import (
    ConfidenceMetadata,
    DepartmentRequest,
    DepartmentResponse,
)

logger = logging.getLogger(__name__)


@dataclass
class Assumption:
    """A hidden assumption extracted from content."""
    text: str
    implicit: bool  # True if not explicitly stated
    confidence: float
    challenged: bool = False


@dataclass
class Attack:
    """An adversarial attack on a claim or assumption."""
    target: str  # What we're attacking
    attack_type: str  # edge_case, contradiction, ambiguity, missing_context
    attack_text: str
    severity: float  # 0-1, how serious the attack is


@dataclass
class Vulnerability:
    """A discovered vulnerability in reasoning."""
    attack: Attack
    confirmed: bool
    impact: str
    mitigation: str | None = None


class RefuseLoom(BaseLoom):
    """
    The Skeptic - Adversarial probing, breaking-grounded.

    Tries to find what's wrong. Excels at finding edge cases,
    hidden assumptions, logical gaps, and stress-testing claims.

    Attributes:
        attack_intensity: How aggressively to probe (default: 0.7)
        assumption_depth: How deep to look for assumptions (default: 3)
        min_attack_severity: Minimum severity to report (default: 0.3)
    """

    # Types of adversarial attacks
    ATTACK_TYPES = [
        "edge_case",        # Extreme inputs/conditions
        "contradiction",    # Logical contradictions
        "ambiguity",        # Unclear meanings
        "missing_context",  # Assumed context not provided
        "false_premise",    # Underlying assumption is wrong
        "scope_creep",      # Claim is too broad
        "cherry_picking",   # Selective evidence
        "straw_man",        # Misrepresenting the argument
    ]

    def __init__(
        self,
        attack_intensity: float = 0.7,
        assumption_depth: int = 3,
        min_attack_severity: float = 0.3,
        attack_backend: Any | None = None,
        **kwargs
    ):
        """
        Initialize the Refuse Loom.

        Args:
            attack_intensity: How aggressively to probe (0-1)
            assumption_depth: How deep to look for hidden assumptions
            min_attack_severity: Minimum severity threshold for attacks
            attack_backend: Optional backend for advanced attacks
            **kwargs: Arguments passed to BaseLoom
        """
        super().__init__(**kwargs)

        # Cognitive style parameters
        self.attack_intensity = attack_intensity
        self.assumption_depth = assumption_depth
        self.min_attack_severity = min_attack_severity

        # Attack backend
        self._attacker = attack_backend

        # Track vulnerabilities for learning
        self._vulnerabilities: list[Vulnerability] = []

    @property
    def perspective(self) -> str:
        """The Skeptic's perspective."""
        return REFUSE

    @property
    def blind_spots(self) -> list[str]:
        """What the Skeptic cannot see."""
        return [
            "constructive solutions",
            "acknowledging what works",
            "nuance and context",
            "good faith interpretation",
            "when criticism is unhelpful",
            "positive aspects of arguments",
        ]

    def admits_ignorance(self, query: str) -> list[str]:
        """
        What the Skeptic knows it doesn't know about this query.

        Args:
            query: The query to assess

        Returns:
            List of specific knowledge gaps
        """
        ignorance = []

        # Check for solution-seeking keywords
        solution_words = ['how to', 'solution', 'fix', 'solve', 'help']
        if any(word in query.lower() for word in solution_words):
            ignorance.append("Cannot provide solutions - only find problems")

        # Check for positive keywords
        positive_words = ['good', 'best', 'positive', 'benefit', 'advantage']
        if any(word in query.lower() for word in positive_words):
            ignorance.append("Cannot identify strengths - only weaknesses")

        # Check for nuanced keywords
        nuance_words = ['depends', 'context', 'sometimes', 'often']
        if any(word in query.lower() for word in nuance_words):
            ignorance.append("May miss contextual nuance in critique")

        if not ignorance:
            ignorance.append("May be unfairly critical of valid arguments")

        return ignorance

    def falsifiable_by(self, claim: str) -> list[str]:
        """
        What evidence would change the Skeptic's claim.

        Args:
            claim: The claim to assess

        Returns:
            List of falsifiers
        """
        return [
            "Showing the attack doesn't apply to this case",
            "Providing missing context that resolves the issue",
            "Demonstrating the assumption is actually valid",
            "Proving the edge case is impossible",
            "Finding flaws in the attack methodology",
        ]

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Execute adversarial probing logic.

        Args:
            request: Department request with query

        Returns:
            Response with attack results
        """
        start_time = datetime.now()
        query = request.query

        # Step 1: Extract claims and assumptions
        claims = self._extract_claims(query)
        assumptions = self._extract_assumptions(query, claims)

        # Step 2: Generate adversarial attacks
        attacks = self._generate_attacks(claims, assumptions)

        # Step 3: Execute attacks
        vulnerabilities = await self._execute_attacks(attacks)
        self._vulnerabilities = vulnerabilities

        # Step 4: Rank by severity
        vulnerabilities.sort(key=lambda v: v.attack.severity, reverse=True)

        # Step 5: Format attack report
        response = self._format_attack_report(vulnerabilities, assumptions)

        # Step 6: Compute confidence (in our attacks)
        confidence = self._compute_attack_confidence(vulnerabilities)

        # Step 7: Compute epistemic confidence
        epistemic = self._compute_epistemic_confidence(vulnerabilities, assumptions)

        # Calculate duration
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Update metrics
        self._update_metrics(confidence, duration_ms)

        # Log discoveries for dreaming
        for vuln in vulnerabilities[:2]:  # Top 2
            if vuln.attack.severity > 0.7:
                self.log_discovery_insight({
                    "attack_type": vuln.attack.attack_type,
                    "target": vuln.attack.target[:50],
                    "severity": vuln.attack.severity,
                }, confidence=vuln.attack.severity)

        # Create fabric
        fabric = self._create_fabric(
            query=query,
            response=response,
            tool_used="refuse",
            confidence=confidence,
            duration_ms=duration_ms,
            epistemic_confidence=epistemic,
            admits_ignorance=self.admits_ignorance(query),
            falsifiable_by=self.falsifiable_by(response),
            metadata={
                "claims_found": len(claims),
                "assumptions_found": len(assumptions),
                "attacks_generated": len(attacks),
                "vulnerabilities_found": len(vulnerabilities),
                "avg_severity": sum(v.attack.severity for v in vulnerabilities) / max(len(vulnerabilities), 1),
            },
        )

        return DepartmentResponse(
            task_id=getattr(request, 'task_id', uuid4()),
            result=response,
            confidence=ConfidenceMetadata.from_score(
                confidence,
                justification=[f"Found {len(vulnerabilities)} vulnerabilities in {len(attacks)} attacks"],
                sources=["refuse_loom"],
            ),
            metadata={
                "perspective": self.perspective,
                "fabric": fabric,
                "attacks": attacks,
                "vulnerabilities": vulnerabilities,
                "assumptions": assumptions,
            },
        )

    def _extract_claims(self, text: str) -> list[str]:
        """
        Extract claims from text.

        Args:
            text: Text to extract from

        Returns:
            List of claim strings
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 5:
                claims.append(sentence)

        return claims

    def _extract_assumptions(
        self,
        text: str,
        claims: list[str]
    ) -> list[Assumption]:
        """
        Extract hidden assumptions from text and claims.

        Args:
            text: Original text
            claims: Extracted claims

        Returns:
            List of assumptions
        """
        assumptions = []

        # Look for implicit assumptions in each claim
        for claim in claims:
            # Pattern: "X because Y" assumes Y is true
            because_match = re.search(r'because\s+(.+)', claim.lower())
            if because_match:
                assumptions.append(Assumption(
                    text=f"Assumes: {because_match.group(1)}",
                    implicit=True,
                    confidence=0.7,
                ))

            # Pattern: "If X then Y" assumes X can happen
            if_match = re.search(r'if\s+(.+?)\s+then', claim.lower())
            if if_match:
                assumptions.append(Assumption(
                    text=f"Assumes possible: {if_match.group(1)}",
                    implicit=True,
                    confidence=0.6,
                ))

            # Pattern: "The X" assumes X exists
            the_match = re.findall(r'\bthe\s+(\w+)', claim.lower())
            for match in the_match[:2]:  # Limit to 2
                if match not in {'way', 'time', 'result', 'answer'}:
                    assumptions.append(Assumption(
                        text=f"Assumes exists: {match}",
                        implicit=True,
                        confidence=0.5,
                    ))

        # Look for domain assumptions
        domain_assumptions = self._find_domain_assumptions(text)
        assumptions.extend(domain_assumptions)

        return assumptions[:self.assumption_depth * 3]  # Limit based on depth

    def _find_domain_assumptions(self, text: str) -> list[Assumption]:
        """
        Find domain-specific assumptions.

        Args:
            text: Text to analyze

        Returns:
            List of domain assumptions
        """
        assumptions = []
        text_lower = text.lower()

        # Technical domain
        if any(w in text_lower for w in ['algorithm', 'code', 'function', 'system']):
            assumptions.append(Assumption(
                text="Assumes technical context and knowledge",
                implicit=True,
                confidence=0.8,
            ))

        # Statistical domain
        if any(w in text_lower for w in ['average', 'probability', 'distribution', 'sample']):
            assumptions.append(Assumption(
                text="Assumes statistical validity of approach",
                implicit=True,
                confidence=0.7,
            ))

        # Temporal domain
        if any(w in text_lower for w in ['always', 'never', 'will', 'must']):
            assumptions.append(Assumption(
                text="Assumes temporal universality",
                implicit=True,
                confidence=0.6,
            ))

        return assumptions

    def _generate_attacks(
        self,
        claims: list[str],
        assumptions: list[Assumption]
    ) -> list[Attack]:
        """
        Generate adversarial attacks.

        Args:
            claims: Claims to attack
            assumptions: Assumptions to attack

        Returns:
            List of attacks
        """
        attacks = []

        # Attack each claim
        for claim in claims:
            claim_attacks = self._generate_claim_attacks(claim)
            attacks.extend(claim_attacks)

        # Attack each assumption
        for assumption in assumptions:
            if assumption.implicit:  # Implicit assumptions are more vulnerable
                attacks.append(Attack(
                    target=assumption.text,
                    attack_type="false_premise",
                    attack_text=f"What if {assumption.text.lower()} is wrong?",
                    severity=0.6 * self.attack_intensity,
                ))

        # Filter by minimum severity
        attacks = [a for a in attacks if a.severity >= self.min_attack_severity]

        return attacks

    def _generate_claim_attacks(self, claim: str) -> list[Attack]:
        """
        Generate attacks for a specific claim.

        Args:
            claim: Claim to attack

        Returns:
            List of attacks
        """
        attacks = []
        claim_lower = claim.lower()

        # Edge case attack
        if any(w in claim_lower for w in ['all', 'every', 'always', 'never']):
            attacks.append(Attack(
                target=claim,
                attack_type="edge_case",
                attack_text="What about edge cases where this doesn't hold?",
                severity=0.7 * self.attack_intensity,
            ))

        # Ambiguity attack
        vague_words = ['some', 'many', 'often', 'usually', 'good', 'bad']
        if any(w in claim_lower for w in vague_words):
            attacks.append(Attack(
                target=claim,
                attack_type="ambiguity",
                attack_text="This is too vague - what exactly do you mean?",
                severity=0.5 * self.attack_intensity,
            ))

        # Missing context attack
        if len(claim.split()) < 15:  # Short claims often lack context
            attacks.append(Attack(
                target=claim,
                attack_type="missing_context",
                attack_text="Under what conditions is this true?",
                severity=0.4 * self.attack_intensity,
            ))

        # Scope creep attack
        broad_words = ['everything', 'nothing', 'everyone', 'no one']
        if any(w in claim_lower for w in broad_words):
            attacks.append(Attack(
                target=claim,
                attack_type="scope_creep",
                attack_text="This claim is too broad - can you narrow it?",
                severity=0.6 * self.attack_intensity,
            ))

        # Contradiction attack (self-referential)
        if 'not' in claim_lower and any(w in claim_lower for w in ['is', 'are', 'can']):
            attacks.append(Attack(
                target=claim,
                attack_type="contradiction",
                attack_text="Does this contradict other claims or itself?",
                severity=0.5 * self.attack_intensity,
            ))

        return attacks

    async def _execute_attacks(
        self,
        attacks: list[Attack]
    ) -> list[Vulnerability]:
        """
        Execute attacks and determine vulnerabilities.

        Args:
            attacks: Attacks to execute

        Returns:
            List of discovered vulnerabilities
        """
        vulnerabilities = []

        for attack in attacks:
            # Simple vulnerability assessment
            # In production, this would use LLM or external verification
            confirmed = self._assess_attack(attack)

            if confirmed:
                vulnerabilities.append(Vulnerability(
                    attack=attack,
                    confirmed=True,
                    impact=self._assess_impact(attack),
                    mitigation=self._suggest_mitigation(attack),
                ))

        return vulnerabilities

    def _assess_attack(self, attack: Attack) -> bool:
        """
        Assess if an attack reveals a real vulnerability.

        Args:
            attack: Attack to assess

        Returns:
            True if attack reveals vulnerability
        """
        # Base confirmation rate depends on attack type
        confirmation_rates = {
            "edge_case": 0.6,
            "contradiction": 0.5,
            "ambiguity": 0.7,
            "missing_context": 0.8,
            "false_premise": 0.4,
            "scope_creep": 0.5,
            "cherry_picking": 0.4,
            "straw_man": 0.3,
        }

        rate = confirmation_rates.get(attack.attack_type, 0.5)

        # Adjust by severity and intensity
        rate = rate * self.attack_intensity

        # Probabilistic confirmation
        return random.random() < rate

    def _assess_impact(self, attack: Attack) -> str:
        """
        Assess the impact of a confirmed vulnerability.

        Args:
            attack: The successful attack

        Returns:
            Impact description
        """
        if attack.severity >= 0.7:
            return "HIGH: Significantly undermines the argument"
        elif attack.severity >= 0.5:
            return "MEDIUM: Weakens but doesn't invalidate"
        else:
            return "LOW: Minor issue, may be acceptable"

    def _suggest_mitigation(self, attack: Attack) -> str:
        """
        Suggest how to mitigate the vulnerability.

        Args:
            attack: The successful attack

        Returns:
            Mitigation suggestion
        """
        mitigations = {
            "edge_case": "Add explicit boundary conditions",
            "contradiction": "Resolve logical inconsistency",
            "ambiguity": "Use more precise language",
            "missing_context": "Provide explicit context/scope",
            "false_premise": "Validate or state assumptions explicitly",
            "scope_creep": "Narrow claim to specific cases",
            "cherry_picking": "Consider counterexamples",
            "straw_man": "Address the actual argument",
        }

        return mitigations.get(attack.attack_type, "Review and revise")

    def _format_attack_report(
        self,
        vulnerabilities: list[Vulnerability],
        assumptions: list[Assumption]
    ) -> str:
        """
        Format attack results as report.

        Args:
            vulnerabilities: Discovered vulnerabilities
            assumptions: Hidden assumptions

        Returns:
            Formatted report text
        """
        parts = ["Adversarial Analysis:\n"]

        # Summary
        if not vulnerabilities:
            parts.append("No significant vulnerabilities found.")
            parts.append("\n[NOTE: This doesn't mean the argument is perfect - "
                        "only that these attacks didn't find issues.]")
            return "\n".join(parts)

        parts.append(f"Found {len(vulnerabilities)} vulnerabilities:\n")

        # Top vulnerabilities
        for i, vuln in enumerate(vulnerabilities[:5], 1):
            parts.append(f"{i}. [{vuln.attack.attack_type.upper()}] (severity: {vuln.attack.severity:.2f})")
            parts.append(f"   Target: {vuln.attack.target[:50]}...")
            parts.append(f"   Attack: {vuln.attack.attack_text}")
            parts.append(f"   Impact: {vuln.impact}")
            if vuln.mitigation:
                parts.append(f"   Fix: {vuln.mitigation}")
            parts.append("")

        # Hidden assumptions warning
        implicit = [a for a in assumptions if a.implicit]
        if implicit:
            parts.append(f"\n⚠ {len(implicit)} Hidden Assumptions Detected:")
            for assumption in implicit[:3]:
                parts.append(f"   - {assumption.text}")

        parts.append("\n[NOTE: The Skeptic looks for problems, not solutions.]")

        return "\n".join(parts)

    def _compute_attack_confidence(
        self,
        vulnerabilities: list[Vulnerability]
    ) -> float:
        """
        Compute confidence in attack results.

        Args:
            vulnerabilities: Discovered vulnerabilities

        Returns:
            Confidence score (0-1)
        """
        if not vulnerabilities:
            return 0.4  # Low confidence when no issues found (might have missed)

        # Confidence based on severity and confirmation
        avg_severity = sum(v.attack.severity for v in vulnerabilities) / len(vulnerabilities)
        confirmation_rate = sum(1 for v in vulnerabilities if v.confirmed) / len(vulnerabilities)

        # Combine: higher severity + confirmation = higher confidence
        confidence = 0.5 * avg_severity + 0.5 * confirmation_rate

        return min(confidence + 0.2, 0.9)  # Cap at 0.9 - never fully certain

    def _compute_epistemic_confidence(
        self,
        vulnerabilities: list[Vulnerability],
        assumptions: list[Assumption]
    ) -> float:
        """
        Compute epistemic confidence.

        Args:
            vulnerabilities: Discovered vulnerabilities
            assumptions: Found assumptions

        Returns:
            Epistemic confidence (0-1)
        """
        # Skeptic has moderate epistemic confidence
        # (good at finding problems, but may miss or over-attack)
        epistemic = 0.7

        # Reduce if we found many implicit assumptions (more uncertainty)
        implicit_count = sum(1 for a in assumptions if a.implicit)
        epistemic -= implicit_count * 0.05

        # Reduce if attacks are all low severity (might be nitpicking)
        if vulnerabilities:
            avg_severity = sum(v.attack.severity for v in vulnerabilities) / len(vulnerabilities)
            if avg_severity < 0.4:
                epistemic -= 0.1

        # Increase if we found high-severity vulnerabilities (strong evidence)
        high_severity = [v for v in vulnerabilities if v.attack.severity >= 0.7]
        if high_severity:
            epistemic += 0.1

        return max(min(epistemic, 0.85), 0.35)


__all__ = ['RefuseLoom']
