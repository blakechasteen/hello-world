"""
Full Odyssey Matryoshka MCTS Integration
Process the entire epic through Bayesian Matryoshka MCTS with temporal analysis
"""

import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time

from full_odyssey_structure import (
    FullOdysseyStructure, NarrativeDecision, DecisionType, 
    NarrativeArc, CharacterState
)
from temporal_bayesian_evolution import TemporalBayesianEvolution, BayesianPrior
from bayesian_all_the_way_down_mcts import (
    BayesianMatryoshkaMCTS, BayesianMatryoshkaNode, 
    MatryoshkaLevel
)

class NarrativeComplexity(Enum):
    """Complexity levels specific to narrative analysis"""
    SIMPLE_INTERACTION = 1    # Basic character interactions
    MORAL_DILEMMA = 2        # Ethical choices
    STRATEGIC_DECISION = 3    # Tactical/strategic thinking
    DIVINE_INTERACTION = 4    # Dealing with gods/fate
    IDENTITY_CRISIS = 5       # Questions of identity/recognition
    TEMPORAL_PARADOX = 6      # Complex time/consequence relationships

class NodeType(Enum):
    """Node types for narrative MCTS"""
    DECISION = "decision"
    OUTCOME = "outcome"
    STATE = "state"

@dataclass
class NarrativeContext:
    """Context for a narrative decision in the Odyssey"""
    current_book: int
    narrative_arc: NarrativeArc
    character_states: Dict[str, CharacterState]
    active_themes: List[str]
    temporal_layers: List[str]
    divine_presence: float
    emotional_intensity: float
    strategic_importance: float

class OdysseyMatryoshkaMCTS:
    """Matryoshka MCTS specifically designed for Odyssey narrative analysis"""
    
    def __init__(self, odyssey_structure: FullOdysseyStructure, 
                 bayesian_evolution: TemporalBayesianEvolution):
        self.odyssey = odyssey_structure
        self.bayesian_evolution = bayesian_evolution
        self.narrative_memory = {}  # Store decision outcomes and learning
        
    def assess_narrative_complexity(self, decision: NarrativeDecision, 
                                  context: NarrativeContext) -> MatryoshkaLevel:
        """Assess the Matryoshka complexity level needed for this narrative decision"""
        
        complexity_score = 0.0
        
        # Base complexity from decision type
        complexity_map = {
            DecisionType.STRATEGIC: 3,
            DecisionType.MORAL: 2, 
            DecisionType.IDENTITY: 5,
            DecisionType.LOYALTY: 2,
            DecisionType.DIVINE: 4,
            DecisionType.HEROIC: 3
        }
        complexity_score += complexity_map.get(decision.decision_type, 2)
        
        # Narrative arc influences complexity
        arc_complexity = {
            NarrativeArc.INVOCATION: 1,
            NarrativeArc.DEPARTURE: 2,
            NarrativeArc.TRIALS: 4,
            NarrativeArc.WANDERING: 3,
            NarrativeArc.RECOGNITION: 5,
            NarrativeArc.RESOLUTION: 6
        }
        complexity_score += arc_complexity.get(decision.narrative_arc, 3)
        
        # Additional factors
        complexity_score += decision.divine_involvement * 2
        complexity_score += decision.temporal_complexity * 2
        complexity_score += decision.narrative_weight * 1
        
        # Emotional and strategic context
        complexity_score += context.emotional_intensity * 1
        complexity_score += context.strategic_importance * 1
        
        # Map to Matryoshka levels
        if complexity_score <= 3:
            return MatryoshkaLevel.THOMPSON_CORE
        elif complexity_score <= 5:
            return MatryoshkaLevel.VARIATIONAL_BAYES
        elif complexity_score <= 7:
            return MatryoshkaLevel.HIERARCHICAL_BAYES
        elif complexity_score <= 9:
            return MatryoshkaLevel.GAUSSIAN_PROCESSES
        elif complexity_score <= 11:
            return MatryoshkaLevel.NEURAL_BAYES
        else:
            return MatryoshkaLevel.NON_PARAMETRIC
            
    async def build_narrative_context(self, decision: NarrativeDecision) -> NarrativeContext:
        """Build rich context for a narrative decision"""
        
        book = self.odyssey.books[decision.book]
        
        # Get character states at this point
        character_states = {}
        for char_name, char_state in book.character_developments.items():
            character_states[char_name] = char_state
            
        # Calculate contextual factors
        divine_presence = decision.divine_involvement
        emotional_intensity = len([theme for theme in book.key_themes 
                                 if 'love' in theme or 'grief' in theme or 'anger' in theme]) / len(book.key_themes)
        strategic_importance = decision.narrative_weight * decision.wisdom_level
        
        return NarrativeContext(
            current_book=decision.book,
            narrative_arc=decision.narrative_arc,
            character_states=character_states,
            active_themes=book.key_themes,
            temporal_layers=getattr(book, 'temporal_setting', '').split(','),
            divine_presence=divine_presence,
            emotional_intensity=emotional_intensity,
            strategic_importance=strategic_importance
        )
        
    async def create_narrative_mcts_node(self, decision: NarrativeDecision, 
                                       context: NarrativeContext) -> BayesianMatryoshkaNode:
        """Create an MCTS node for narrative decision analysis"""
        
        complexity_level = self.assess_narrative_complexity(decision, context)
        
        # Build state representation
        state = {
            "decision": decision.description,
            "options": decision.options,
            "canonical_choice": decision.canonical_choice,
            "narrative_arc": decision.narrative_arc.value,
            "book": decision.book,
            "characters": list(context.character_states.keys()),
            "themes": context.active_themes,
            "divine_involvement": decision.divine_involvement,
            "temporal_complexity": decision.temporal_complexity
        }
        
        # Create Matryoshka node
        node = BayesianMatryoshkaNode(
            state=state,
            complexity_level=complexity_level,
            node_type=NodeType.DECISION,
            parent=None
        )
        
        # Add narrative-specific belief tracking
        node.narrative_beliefs = {
            "outcome_quality": self.bayesian_evolution.priors["wisdom_over_force"].current_value,
            "divine_favor": self.bayesian_evolution.priors["divine_favor"].current_value,
            "long_term_consequences": self.bayesian_evolution.priors["future_consequences_predictable"].current_value
        }
        
        return node
        
    async def evaluate_narrative_outcome(self, node: BayesianMatryoshkaNode, 
                                       decision: NarrativeDecision) -> float:
        """Evaluate the quality of a narrative decision outcome"""
        
        # Base evaluation from wisdom and narrative weight
        base_score = (decision.wisdom_level * 0.6 + decision.narrative_weight * 0.4)
        
        # Adjust based on current Bayesian beliefs
        belief_adjustment = 0.0
        
        # Divine decisions should align with divine favor beliefs
        if decision.decision_type == DecisionType.DIVINE:
            divine_favor = self.bayesian_evolution.priors["divine_favor"].current_value
            if decision.divine_involvement > 0.5:
                belief_adjustment += divine_favor * 0.2
                
        # Strategic decisions should align with wisdom beliefs
        if decision.decision_type == DecisionType.STRATEGIC:
            wisdom_belief = self.bayesian_evolution.priors["wisdom_over_force"].current_value
            if decision.wisdom_level > 0.6:
                belief_adjustment += wisdom_belief * 0.3
                
        # Identity decisions are risky based on learned beliefs
        if decision.decision_type == DecisionType.IDENTITY:
            identity_safety = self.bayesian_evolution.priors["identity_revelation_safe"].current_value
            revelation_risk = 1.0 - identity_safety
            if "reveal" in decision.canonical_choice:
                belief_adjustment -= revelation_risk * 0.2
                
        # Temporal complexity affects predictability
        if decision.temporal_complexity > 0.7:
            predictability = self.bayesian_evolution.priors["future_consequences_predictable"].current_value
            belief_adjustment += predictability * 0.1
            
        final_score = base_score + belief_adjustment
        
        # Add Matryoshka-level specific refinements
        if hasattr(node, 'complexity_level'):
            if node.complexity_level.value >= MatryoshkaLevel.NEURAL_BAYES.value:
                # Higher complexity levels get more nuanced evaluation
                narrative_nuance = np.random.beta(2, 2) * 0.1  # Small but meaningful adjustment
                final_score += narrative_nuance
                
        return np.clip(final_score, 0.0, 1.0)
        
    async def run_narrative_mcts(self, decision: NarrativeDecision, 
                               simulations: int = 100) -> Dict[str, Any]:
        """Run MCTS analysis on a narrative decision"""
        
        start_time = time.time()
        
        # Build context
        context = await self.build_narrative_context(decision)
        
        # Create root node
        root = await self.create_narrative_mcts_node(decision, context)
        
        # Initialize MCTS
        mcts = BayesianMatryoshkaMCTS(
            exploration_constant=1.414,
            max_depth=5,
            time_limit=2.0  # 2 second limit per decision
        )
        
        # Run simulations
        simulation_results = []
        
        for sim in range(simulations):
            # Select phase
            selected_node = await mcts.select(root)
            
            # Expand phase - create children for decision options
            if not selected_node.children and decision.options:
                for option in decision.options[:3]:  # Limit to top 3 options
                    child_state = selected_node.state.copy()
                    child_state["chosen_option"] = option
                    
                    child = BayesianMatryoshkaNode(
                        state=child_state,
                        complexity_level=selected_node.complexity_level,
                        node_type=NodeType.OUTCOME,
                        parent=selected_node
                    )
                    selected_node.children.append(child)
                    
            # Simulate phase
            if selected_node.children:
                chosen_child = np.random.choice(selected_node.children)
                outcome_quality = await self.evaluate_narrative_outcome(chosen_child, decision)
                
                # Backpropagate
                await mcts.backpropagate(chosen_child, outcome_quality)
                
                simulation_results.append({
                    "option": chosen_child.state.get("chosen_option", "unknown"),
                    "quality": outcome_quality,
                    "complexity_level": chosen_child.complexity_level.value
                })
                
        # Analyze results
        analysis_time = time.time() - start_time
        
        # Find best options
        option_scores = {}
        for result in simulation_results:
            option = result["option"]
            if option not in option_scores:
                option_scores[option] = []
            option_scores[option].append(result["quality"])
            
        best_options = []
        for option, scores in option_scores.items():
            avg_score = np.mean(scores)
            confidence = 1.0 - np.std(scores)  # Higher std = lower confidence
            best_options.append({
                "option": option,
                "avg_score": avg_score,
                "confidence": confidence,
                "simulations": len(scores)
            })
            
        best_options.sort(key=lambda x: x["avg_score"], reverse=True)
        
        return {
            "decision": decision.description,
            "canonical_choice": decision.canonical_choice,
            "complexity_level": root.complexity_level.name,
            "best_options": best_options,
            "total_simulations": simulations,
            "analysis_time": analysis_time,
            "context": {
                "narrative_arc": context.narrative_arc.value,
                "divine_presence": context.divine_presence,
                "strategic_importance": context.strategic_importance
            }
        }

class FullOdysseyProcessor:
    """Process the complete Odyssey through Matryoshka MCTS"""
    
    def __init__(self):
        self.odyssey = FullOdysseyStructure()
        self.bayesian_evolution = TemporalBayesianEvolution(self.odyssey)
        self.narrative_mcts = OdysseyMatryoshkaMCTS(self.odyssey, self.bayesian_evolution)
        self.processing_results = []
        
    async def process_full_epic(self, max_decisions: int = 10) -> Dict[str, Any]:
        """Process the full Odyssey through Matryoshka MCTS"""
        
        print("🏛️  PROCESSING FULL ODYSSEY THROUGH MATRYOSHKA MCTS")
        print("=" * 70)
        
        # Get all decisions in chronological order
        all_decisions = self.odyssey.get_decision_sequence()
        decisions_to_process = all_decisions[:max_decisions]  # Limit for performance
        
        print(f"Processing {len(decisions_to_process)} key decisions across the epic...")
        print()
        
        for i, decision in enumerate(decisions_to_process):
            print(f"📖 {i+1}/{len(decisions_to_process)} - Book {decision.book}: {decision.scene}")
            
            # Run MCTS analysis
            mcts_result = await self.narrative_mcts.run_narrative_mcts(decision, simulations=50)
            
            # Update Bayesian beliefs based on this decision
            belief_updates = await self.bayesian_evolution.process_decision(decision)
            
            # Store results
            processing_result = {
                "decision_index": i,
                "book": decision.book,
                "mcts_analysis": mcts_result,
                "belief_updates": belief_updates,
                "canonical_vs_optimal": self._compare_canonical_vs_optimal(decision, mcts_result)
            }
            
            self.processing_results.append(processing_result)
            
            # Show key insights
            best_option = mcts_result["best_options"][0] if mcts_result["best_options"] else None
            if best_option:
                print(f"   MCTS Optimal: {best_option['option']} (Score: {best_option['avg_score']:.3f})")
                print(f"   Canonical: {decision.canonical_choice}")
                print(f"   Complexity: {mcts_result['complexity_level']}")
                
                if belief_updates["prior_updates"]:
                    print(f"   Belief Update: {belief_updates['prior_updates'][0]}")
                    
            print()
            
        return await self._generate_epic_analysis()
        
    def _compare_canonical_vs_optimal(self, decision: NarrativeDecision, 
                                    mcts_result: Dict) -> Dict[str, Any]:
        """Compare canonical choice vs MCTS optimal"""
        
        canonical = decision.canonical_choice
        best_options = mcts_result["best_options"]
        
        canonical_score = None
        optimal_score = best_options[0]["avg_score"] if best_options else 0
        optimal_choice = best_options[0]["option"] if best_options else "unknown"
        
        # Find canonical choice in results
        for option_data in best_options:
            if canonical in option_data["option"] or option_data["option"] in canonical:
                canonical_score = option_data["avg_score"]
                break
                
        return {
            "canonical_choice": canonical,
            "canonical_score": canonical_score,
            "optimal_choice": optimal_choice,
            "optimal_score": optimal_score,
            "improvement_possible": (optimal_score - canonical_score) if canonical_score else None
        }
        
    async def _generate_epic_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis of the full epic processing"""
        
        print("\n🔍 FULL EPIC ANALYSIS")
        print("=" * 50)
        
        # Complexity distribution
        complexity_counts = {}
        for result in self.processing_results:
            complexity = result["mcts_analysis"]["complexity_level"]
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
            
        print("Complexity Distribution:")
        for complexity, count in complexity_counts.items():
            print(f"  {complexity}: {count} decisions")
            
        # Canonical vs optimal analysis
        improvements = []
        canonical_optimal_matches = 0
        
        for result in self.processing_results:
            comparison = result["canonical_vs_optimal"]
            if comparison["improvement_possible"] is not None:
                improvements.append(comparison["improvement_possible"])
                if abs(comparison["improvement_possible"]) < 0.1:
                    canonical_optimal_matches += 1
                    
        if improvements:
            avg_improvement = np.mean(improvements)
            max_improvement = max(improvements)
            print(f"\nCanonical vs Optimal Analysis:")
            print(f"  Average improvement possible: {avg_improvement:.3f}")
            print(f"  Maximum improvement possible: {max_improvement:.3f}")
            print(f"  Canonical-optimal matches: {canonical_optimal_matches}/{len(improvements)}")
            
        # Belief evolution summary
        evolution_summary = self.bayesian_evolution.get_temporal_evolution_summary()
        print(f"\nBelief Evolution Summary:")
        
        significant_changes = []
        for name, data in evolution_summary["prior_evolution"].items():
            if abs(data["change"]) > 0.2:  # Significant changes
                significant_changes.append((name, data["change"]))
                
        significant_changes.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, change in significant_changes[:5]:  # Top 5 changes
            print(f"  {name}: {change:+.3f} change")
            
        return {
            "total_decisions_processed": len(self.processing_results),
            "complexity_distribution": complexity_counts,
            "average_improvement_possible": np.mean(improvements) if improvements else 0,
            "canonical_optimal_match_rate": canonical_optimal_matches / len(improvements) if improvements else 0,
            "significant_belief_changes": significant_changes,
            "processing_results": self.processing_results
        }

async def test_full_odyssey_integration():
    """Test the full Odyssey Matryoshka MCTS integration"""
    
    processor = FullOdysseyProcessor()
    results = await processor.process_full_epic(max_decisions=8)  # Process 8 key decisions
    
    print("\n✨ FULL ODYSSEY MATRYOSHKA MCTS COMPLETE!")
    print(f"📊 Processed {results['total_decisions_processed']} decisions")
    print(f"🎯 Canonical-optimal match rate: {results['canonical_optimal_match_rate']:.1%}")
    print(f"📈 Average improvement possible: {results['average_improvement_possible']:.3f}")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_full_odyssey_integration())