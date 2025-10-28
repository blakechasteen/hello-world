"""
Temporal Bayesian Evolution System
Track how Bayesian priors evolve across the Odyssey narrative sequence
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio

from full_odyssey_structure import (
    FullOdysseyStructure, NarrativeDecision, DecisionType, 
    NarrativeArc, CharacterState
)

@dataclass
class BayesianPrior:
    """A Bayesian prior belief that evolves over time"""
    belief_name: str
    initial_value: float
    current_value: float
    confidence: float = 0.5
    evidence_count: int = 0
    update_history: List[Tuple[int, float, str]] = field(default_factory=list)  # (book, new_value, reason)
    
    def update(self, book_num: int, evidence: float, reason: str, learning_rate: float = 0.1):
        """Bayesian update based on new evidence"""
        # Simple Bayesian update with confidence weighting
        weight = learning_rate * (1.0 + self.confidence)
        old_value = self.current_value
        self.current_value = (1 - weight) * self.current_value + weight * evidence
        
        # Update confidence based on consistency
        consistency = 1.0 - abs(evidence - old_value)
        self.confidence = 0.9 * self.confidence + 0.1 * consistency
        
        self.evidence_count += 1
        self.update_history.append((book_num, self.current_value, reason))

@dataclass 
class StrategicLearning:
    """Track strategic learning across decisions"""
    strategy_type: str
    success_rate: float = 0.5
    total_attempts: int = 0
    successes: int = 0
    recent_performance: List[bool] = field(default_factory=list)
    
    def record_outcome(self, success: bool):
        """Record the outcome of using this strategy"""
        self.total_attempts += 1
        if success:
            self.successes += 1
        
        self.recent_performance.append(success)
        if len(self.recent_performance) > 10:  # Keep recent history
            self.recent_performance.pop(0)
            
        self.success_rate = self.successes / self.total_attempts

class TemporalBayesianEvolution:
    """System to track Bayesian evolution across the Odyssey"""
    
    def __init__(self, odyssey_structure: FullOdysseyStructure):
        self.odyssey = odyssey_structure
        self.priors = self._initialize_priors()
        self.strategic_learning = self._initialize_strategic_learning()
        self.character_beliefs = self._initialize_character_beliefs()
        self.temporal_windows = self._define_temporal_windows()
        
    def _initialize_priors(self) -> Dict[str, BayesianPrior]:
        """Initialize core Bayesian priors that evolve"""
        priors = {}
        
        # Core strategic beliefs
        priors["divine_favor"] = BayesianPrior("divine_favor", 0.7, 0.7, 0.6)
        priors["crew_loyalty"] = BayesianPrior("crew_loyalty", 0.8, 0.8, 0.5)
        priors["home_achievable"] = BayesianPrior("home_achievable", 0.6, 0.6, 0.4)
        priors["wisdom_over_force"] = BayesianPrior("wisdom_over_force", 0.5, 0.5, 0.3)
        priors["identity_revelation_safe"] = BayesianPrior("identity_revelation_safe", 0.4, 0.4, 0.3)
        priors["divine_intervention_likely"] = BayesianPrior("divine_intervention_likely", 0.6, 0.6, 0.5)
        
        # Temporal-based beliefs
        priors["prophecy_accuracy"] = BayesianPrior("prophecy_accuracy", 0.5, 0.5, 0.2)
        priors["past_decisions_matter"] = BayesianPrior("past_decisions_matter", 0.7, 0.7, 0.4)
        priors["future_consequences_predictable"] = BayesianPrior("future_consequences_predictable", 0.3, 0.3, 0.2)
        
        return priors
        
    def _initialize_strategic_learning(self) -> Dict[str, StrategicLearning]:
        """Initialize strategic learning tracking"""
        learning = {}
        
        # Decision strategies that are learned
        learning["hubris_strategy"] = StrategicLearning("hubris_strategy", 0.3)  # Start low
        learning["patience_strategy"] = StrategicLearning("patience_strategy", 0.6)
        learning["deception_strategy"] = StrategicLearning("deception_strategy", 0.5)
        learning["direct_confrontation"] = StrategicLearning("direct_confrontation", 0.4)
        learning["divine_appeal"] = StrategicLearning("divine_appeal", 0.7)
        learning["trust_companions"] = StrategicLearning("trust_companions", 0.6)
        
        return learning
        
    def _initialize_character_beliefs(self) -> Dict[str, Dict[str, BayesianPrior]]:
        """Initialize beliefs about characters"""
        char_beliefs = defaultdict(dict)
        
        # Key relationships with evolving trust
        char_beliefs["Athena"]["trustworthy"] = BayesianPrior("athena_trustworthy", 0.8, 0.8, 0.7)
        char_beliefs["Athena"]["helpful"] = BayesianPrior("athena_helpful", 0.9, 0.9, 0.8)
        
        char_beliefs["Poseidon"]["hostile"] = BayesianPrior("poseidon_hostile", 0.5, 0.5, 0.4)
        char_beliefs["Poseidon"]["implacable"] = BayesianPrior("poseidon_implacable", 0.3, 0.3, 0.2)
        
        char_beliefs["Penelope"]["faithful"] = BayesianPrior("penelope_faithful", 0.7, 0.7, 0.5)
        char_beliefs["Penelope"]["wise"] = BayesianPrior("penelope_wise", 0.6, 0.6, 0.4)
        
        char_beliefs["Crew"]["loyal"] = BayesianPrior("crew_loyal", 0.7, 0.7, 0.5)
        char_beliefs["Crew"]["disciplined"] = BayesianPrior("crew_disciplined", 0.5, 0.5, 0.4)
        
        return char_beliefs
        
    def _define_temporal_windows(self) -> Dict[str, List[int]]:
        """Define temporal analysis windows"""
        return {
            "early_journey": [1, 2, 3, 4, 5],
            "major_trials": [9, 10, 11, 12],
            "wandering_period": [13, 14, 15, 16],
            "homecoming": [17, 18, 19, 20],
            "resolution": [21, 22, 23, 24]
        }
        
    async def process_decision(self, decision: NarrativeDecision) -> Dict[str, Any]:
        """Process a decision and update Bayesian beliefs"""
        updates = {
            "prior_updates": [],
            "strategic_learning": [],
            "character_belief_updates": [],
            "temporal_insights": []
        }
        
        # Update priors based on decision outcomes
        await self._update_priors_from_decision(decision, updates)
        
        # Update strategic learning
        await self._update_strategic_learning(decision, updates)
        
        # Update character beliefs
        await self._update_character_beliefs(decision, updates)
        
        # Analyze temporal patterns
        await self._analyze_temporal_patterns(decision, updates)
        
        return updates
        
    async def _update_priors_from_decision(self, decision: NarrativeDecision, updates: Dict):
        """Update core priors based on decision outcomes"""
        book_num = decision.book
        
        # Divine involvement updates divine-related priors
        if decision.divine_involvement > 0.5:
            evidence = min(decision.divine_involvement + 0.2, 1.0)
            self.priors["divine_intervention_likely"].update(
                book_num, evidence, f"High divine involvement in {decision.scene}"
            )
            updates["prior_updates"].append(f"Divine intervention likelihood increased to {evidence:.3f}")
            
        # Wisdom level affects wisdom vs force beliefs
        if decision.decision_type == DecisionType.STRATEGIC:
            evidence = decision.wisdom_level
            self.priors["wisdom_over_force"].update(
                book_num, evidence, f"Strategic decision in {decision.scene}"
            )
            updates["prior_updates"].append(f"Wisdom over force updated to {evidence:.3f}")
            
        # Identity decisions affect identity revelation safety
        if decision.decision_type == DecisionType.IDENTITY:
            # If high narrative weight and low wisdom, identity revelation was risky
            risk_evidence = 1.0 - (decision.narrative_weight * (1.0 - decision.wisdom_level))
            self.priors["identity_revelation_safe"].update(
                book_num, risk_evidence, f"Identity decision in {decision.scene}"
            )
            updates["prior_updates"].append(f"Identity revelation safety updated to {risk_evidence:.3f}")
            
        # High narrative weight decisions affect belief in consequence predictability
        if decision.narrative_weight > 0.7:
            # Major decisions with clear consequences increase predictability belief
            evidence = decision.narrative_weight * decision.temporal_complexity
            self.priors["future_consequences_predictable"].update(
                book_num, evidence, f"Major decision with clear consequences"
            )
            updates["prior_updates"].append(f"Consequence predictability updated to {evidence:.3f}")
            
    async def _update_strategic_learning(self, decision: NarrativeDecision, updates: Dict):
        """Update strategic learning based on decision outcomes"""
        
        # Map decision characteristics to strategies
        if decision.wisdom_level < 0.4 and decision.narrative_weight > 0.7:
            # Hubris strategy used
            success = decision.wisdom_level > 0.3  # Some minimal success threshold
            self.strategic_learning["hubris_strategy"].record_outcome(success)
            updates["strategic_learning"].append(f"Hubris strategy: {success}")
            
        if decision.wisdom_level > 0.7:
            # Patience/wisdom strategy
            success = decision.narrative_weight > 0.5  # Good outcomes
            self.strategic_learning["patience_strategy"].record_outcome(success)
            updates["strategic_learning"].append(f"Patience strategy: {success}")
            
        if "deception" in decision.description.lower() or "disguise" in decision.description.lower():
            success = decision.wisdom_level > 0.6
            self.strategic_learning["deception_strategy"].record_outcome(success)
            updates["strategic_learning"].append(f"Deception strategy: {success}")
            
    async def _update_character_beliefs(self, decision: NarrativeDecision, updates: Dict):
        """Update beliefs about characters based on decision outcomes"""
        
        for char_name in decision.affected_characters:
            if char_name in self.character_beliefs:
                char_beliefs = self.character_beliefs[char_name]
                
                # Update based on decision context
                if decision.divine_involvement > 0.5 and char_name in ["Athena", "Zeus", "Poseidon"]:
                    if decision.wisdom_level > 0.6:
                        # Good divine outcome increases trust in helpful gods
                        if "helpful" in char_beliefs:
                            evidence = decision.divine_involvement * decision.wisdom_level
                            char_beliefs["helpful"].update(
                                decision.book, evidence, f"Helpful intervention in {decision.scene}"
                            )
                            updates["character_belief_updates"].append(f"{char_name} helpfulness updated")
                            
    async def _analyze_temporal_patterns(self, decision: NarrativeDecision, updates: Dict):
        """Analyze temporal patterns in decision making"""
        
        # Check which temporal window this decision falls into
        current_window = None
        for window_name, books in self.temporal_windows.items():
            if decision.book in books:
                current_window = window_name
                break
                
        if current_window:
            # Analyze how decision patterns change across temporal windows
            temporal_insight = f"Decision in {current_window} window shows "
            
            if decision.temporal_complexity > 0.6:
                temporal_insight += "high temporal complexity"
                # Update belief about past decisions mattering
                evidence = decision.temporal_complexity
                self.priors["past_decisions_matter"].update(
                    decision.book, evidence, f"Complex temporal decision in {current_window}"
                )
            else:
                temporal_insight += "present-focused decision"
                
            updates["temporal_insights"].append(temporal_insight)
            
    def get_temporal_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of how beliefs evolved over time"""
        summary = {
            "prior_evolution": {},
            "strategic_performance": {},
            "character_belief_evolution": {},
            "temporal_windows_analysis": {}
        }
        
        # Prior evolution
        for name, prior in self.priors.items():
            if prior.update_history:
                summary["prior_evolution"][name] = {
                    "initial": prior.initial_value,
                    "final": prior.current_value,
                    "change": prior.current_value - prior.initial_value,
                    "confidence": prior.confidence,
                    "updates": len(prior.update_history)
                }
                
        # Strategic performance
        for name, learning in self.strategic_learning.items():
            if learning.total_attempts > 0:
                summary["strategic_performance"][name] = {
                    "success_rate": learning.success_rate,
                    "attempts": learning.total_attempts,
                    "recent_trend": np.mean(learning.recent_performance[-5:]) if learning.recent_performance else 0
                }
                
        return summary
        
    def analyze_belief_convergence(self) -> Dict[str, float]:
        """Analyze how beliefs converged over time"""
        convergence = {}
        
        for name, prior in self.priors.items():
            if len(prior.update_history) > 1:
                # Calculate variance in recent updates to measure convergence
                recent_values = [update[1] for update in prior.update_history[-5:]]
                variance = np.var(recent_values) if len(recent_values) > 1 else 1.0
                convergence[name] = 1.0 - min(variance, 1.0)  # Higher = more converged
            else:
                convergence[name] = 0.0
                
        return convergence

async def test_temporal_bayesian_evolution():
    """Test the temporal Bayesian evolution system"""
    print("🕒 TEMPORAL BAYESIAN EVOLUTION TEST")
    print("=" * 60)
    
    # Initialize systems
    odyssey = FullOdysseyStructure()
    bayesian_evolution = TemporalBayesianEvolution(odyssey)
    
    # Process a sequence of key decisions
    key_decisions = odyssey.get_decision_sequence()[:6]  # First 6 decisions
    
    print("Processing key decisions chronologically...\n")
    
    for i, decision in enumerate(key_decisions):
        print(f"📖 Book {decision.book}: {decision.scene}")
        print(f"   Decision: {decision.canonical_choice}")
        print(f"   Wisdom: {decision.wisdom_level:.2f}, Weight: {decision.narrative_weight:.2f}")
        
        # Process decision and update beliefs
        updates = await bayesian_evolution.process_decision(decision)
        
        # Show key updates
        if updates["prior_updates"]:
            print(f"   Prior Updates: {updates['prior_updates'][0]}")
        if updates["strategic_learning"]:
            print(f"   Strategic Learning: {updates['strategic_learning'][0]}")
            
        print()
        
    # Show evolution summary
    print("\n🧠 BELIEF EVOLUTION SUMMARY")
    print("-" * 40)
    
    summary = bayesian_evolution.get_temporal_evolution_summary()
    
    print("Key Prior Changes:")
    for name, data in summary["prior_evolution"].items():
        if abs(data["change"]) > 0.1:  # Significant changes
            print(f"  {name}: {data['initial']:.3f} → {data['final']:.3f} (Δ{data['change']:+.3f})")
            
    print("\nStrategic Learning:")
    for name, data in summary["strategic_performance"].items():
        if data["attempts"] > 0:
            print(f"  {name}: {data['success_rate']:.2f} success rate ({data['attempts']} attempts)")
            
    # Convergence analysis
    convergence = bayesian_evolution.analyze_belief_convergence()
    print(f"\nBelief Convergence (higher = more stable):")
    for name, conv_score in convergence.items():
        if conv_score > 0:
            print(f"  {name}: {conv_score:.3f}")
            
    return bayesian_evolution

if __name__ == "__main__":
    asyncio.run(test_temporal_bayesian_evolution())