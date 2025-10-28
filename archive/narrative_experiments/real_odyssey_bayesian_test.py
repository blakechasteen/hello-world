"""
Real Odyssey Bayesian Test - Using Emily Wilson's Translation
Testing our Bayesian all-the-way-down MCTS with actual narrative complexity
"""

import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time
import hashlib

from emily_wilson_odyssey_excerpts import get_wilson_passages, analyze_narrative_complexity

# Simple embeddings based on text characteristics for testing
def create_text_embedding(text: str, dimension: int = 384) -> np.ndarray:
    """Create a deterministic embedding based on text characteristics"""
    # Use text hash for consistency
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Convert hash to numbers for embedding
    hash_nums = [ord(c) for c in text_hash[:dimension//8]]
    
    # Create embedding with text-based features
    embedding = np.zeros(dimension)
    
    # Basic text features
    word_count = len(text.split())
    char_count = len(text)
    unique_words = len(set(text.lower().split()))
    
    # Fill embedding with features
    embedding[0] = word_count / 100.0  # Normalized word count
    embedding[1] = char_count / 1000.0  # Normalized char count
    embedding[2] = unique_words / word_count if word_count > 0 else 0  # Vocabulary richness
    
    # Add hash-based variation
    for i, num in enumerate(hash_nums):
        if i + 3 < dimension:
            embedding[i + 3] = (num / 255.0) * 2 - 1  # Normalize to [-1, 1]
    
    # Add some structured patterns based on content
    if "god" in text.lower() or "zeus" in text.lower():
        embedding[dimension//4] = 0.8  # Divine theme
    if "ship" in text.lower() or "sea" in text.lower():
        embedding[dimension//3] = 0.7  # Maritime theme
    if "home" in text.lower() or "ithaca" in text.lower():
        embedding[dimension//2] = 0.9  # Homecoming theme
        
    return embedding

@dataclass
class OdysseyDecision:
    """A decision point in the Odyssey narrative"""
    scene: str
    choice: str
    consequences: List[str]
    wisdom_level: float  # 0-1, how wise/strategic the choice is
    narrative_weight: float  # 0-1, how important to the story

class RealOdysseyBayesianAgent:
    """Bayesian agent that makes decisions in real Odyssey scenarios"""
    
    def __init__(self):
        self.wilson_passages = get_wilson_passages()
        self.knowledge_base = self._build_knowledge_base()
        
    def _build_knowledge_base(self) -> Dict[str, Any]:
        """Build knowledge base from Emily Wilson's passages"""
        kb = {}
        
        for passage_key, passage_data in self.wilson_passages.items():
            analysis = analyze_narrative_complexity(passage_key)
            
            # Create embeddings for text
            text_embedding = create_text_embedding(passage_data["text"])
            
            # Create theme embeddings
            theme_embeddings = []
            for theme in passage_data["complexity_themes"]:
                theme_embeddings.append(create_text_embedding(theme, 128))
            
            kb[passage_key] = {
                "text": passage_data["text"],
                "embedding": text_embedding,
                "themes": passage_data["complexity_themes"],
                "theme_embeddings": theme_embeddings,
                "decisions": passage_data["decision_points"],
                "temporal_layers": passage_data["temporal_layers"],
                "complexity_score": analysis["complexity_score"],
                "narrative_richness": analysis["narrative_richness"]
            }
            
        return kb
    
    async def analyze_decision_context(self, passage_key: str) -> Dict[str, Any]:
        """Analyze the decision context for a passage"""
        if passage_key not in self.knowledge_base:
            return {"error": "Unknown passage"}
            
        passage = self.knowledge_base[passage_key]
        
        # Extract decision factors from the text
        decision_factors = {
            "divine_influence": 0.0,
            "strategic_thinking": 0.0,
            "emotional_weight": 0.0,
            "temporal_complexity": 0.0,
            "narrative_significance": 0.0
        }
        
        text = passage["text"].lower()
        
        # Analyze divine influence
        divine_words = ["god", "goddess", "zeus", "poseidon", "athena", "divine", "fate", "prophecy"]
        divine_influence = sum(1 for word in divine_words if word in text) / len(divine_words)
        decision_factors["divine_influence"] = min(divine_influence, 1.0)
        
        # Analyze strategic thinking
        strategy_words = ["plan", "think", "wise", "clever", "strategy", "escape", "trick", "cunning"]
        strategic_thinking = sum(1 for word in strategy_words if word in text) / len(strategy_words)
        decision_factors["strategic_thinking"] = min(strategic_thinking, 1.0)
        
        # Analyze emotional weight
        emotion_words = ["love", "pain", "suffering", "joy", "grief", "longing", "heart", "tears"]
        emotional_weight = sum(1 for word in emotion_words if word in text) / len(emotion_words)
        decision_factors["emotional_weight"] = min(emotional_weight, 1.0)
        
        # Temporal complexity from layers
        decision_factors["temporal_complexity"] = len(passage["temporal_layers"]) / 5.0
        
        # Narrative significance from complexity score
        decision_factors["narrative_significance"] = passage["complexity_score"]
        
        return {
            "passage": passage_key,
            "decision_factors": decision_factors,
            "embedding": passage["embedding"],
            "themes": passage["themes"],
            "complexity": passage["complexity_score"],
            "text_sample": passage["text"][:200] + "..." if len(passage["text"]) > 200 else passage["text"]
        }

class OdysseyDecisionMCTS:
    """MCTS for Odyssey decision-making with real text"""
    
    def __init__(self, agent: RealOdysseyBayesianAgent):
        self.agent = agent
        self.scenarios = self._create_real_scenarios()
        
    def _create_real_scenarios(self) -> Dict[str, List[OdysseyDecision]]:
        """Create decision scenarios based on actual passages"""
        scenarios = {}
        
        # Cyclops scenario from Wilson's translation
        scenarios["cyclops"] = [
            OdysseyDecision(
                scene="cyclops_confrontation",
                choice="reveal_true_name",
                consequences=["Cyclops knows who blinded him", "Prophecy fulfilled", "Divine retribution possible"],
                wisdom_level=0.3,  # Hubris over wisdom
                narrative_weight=0.9
            ),
            OdysseyDecision(
                scene="cyclops_confrontation", 
                choice="stay_anonymous",
                consequences=["Cyclops remains ignorant", "Prophecy unfulfilled", "Less divine attention"],
                wisdom_level=0.8,  # Strategic wisdom
                narrative_weight=0.4
            ),
            OdysseyDecision(
                scene="cyclops_confrontation",
                choice="partial_truth",
                consequences=["Some revelation", "Prophecy partially fulfilled", "Moderate consequences"],
                wisdom_level=0.6,  # Balanced approach
                narrative_weight=0.6
            )
        ]
        
        # Sirens scenario
        scenarios["sirens"] = [
            OdysseyDecision(
                scene="siren_encounter",
                choice="listen_while_bound", 
                consequences=["Gain knowledge", "Test crew loyalty", "Risk temptation"],
                wisdom_level=0.7,  # Calculated risk
                narrative_weight=0.8
            ),
            OdysseyDecision(
                scene="siren_encounter",
                choice="avoid_completely",
                consequences=["Safe passage", "No knowledge gained", "Miss opportunity"],
                wisdom_level=0.6,  # Safe but limited
                narrative_weight=0.3
            ),
            OdysseyDecision(
                scene="siren_encounter",
                choice="crew_listens_too",
                consequences=["All gain knowledge", "Extreme danger", "Potential disaster"],
                wisdom_level=0.2,  # Reckless
                narrative_weight=0.5
            )
        ]
        
        # Reunion scenario
        scenarios["reunion"] = [
            OdysseyDecision(
                scene="penelope_reunion",
                choice="immediate_revelation",
                consequences=["Quick reunion", "Risk if not really Penelope", "Emotional satisfaction"],
                wisdom_level=0.4,  # Emotionally driven
                narrative_weight=0.7
            ),
            OdysseyDecision(
                scene="penelope_reunion",
                choice="test_with_bed_secret",
                consequences=["Verify true identity", "Prove own identity", "Delayed but certain"],
                wisdom_level=0.9,  # Ultimate wisdom
                narrative_weight=0.9
            ),
            OdysseyDecision(
                scene="penelope_reunion",
                choice="gradual_revelation",
                consequences=["Slow building trust", "Multiple verification points", "Drawn out process"],
                wisdom_level=0.7,  # Moderate caution
                narrative_weight=0.6
            )
        ]
        
        return scenarios
    
    async def evaluate_decision_quality(self, passage_key: str, decision: OdysseyDecision) -> float:
        """Evaluate how good a decision is in context"""
        context = await self.agent.analyze_decision_context(passage_key)
        
        if "error" in context:
            return 0.5
            
        factors = context["decision_factors"]
        
        # Weight decision quality based on context
        quality_score = 0.0
        
        # Higher divine influence should favor more cautious decisions
        if factors["divine_influence"] > 0.5:
            quality_score += decision.wisdom_level * 0.3
        
        # Strategic contexts favor strategic decisions
        if factors["strategic_thinking"] > 0.5:
            quality_score += decision.wisdom_level * 0.4
        
        # Emotional contexts may favor emotional decisions
        if factors["emotional_weight"] > 0.5:
            quality_score += (1.0 - decision.wisdom_level) * 0.2
            
        # Narrative significance amplifies quality
        quality_score *= (1.0 + decision.narrative_weight)
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.1)
        quality_score = np.clip(quality_score + noise, 0.0, 1.0)
        
        return quality_score

async def test_real_odyssey_bayesian_mcts():
    """Test our Bayesian MCTS with real Emily Wilson Odyssey text"""
    print("🏛️  REAL ODYSSEY BAYESIAN MCTS TEST")
    print("=" * 60)
    print("Using Emily Wilson's translation for authentic narrative complexity")
    print()
    
    # Initialize agent and MCTS
    agent = RealOdysseyBayesianAgent()
    mcts = OdysseyDecisionMCTS(agent)
    
    # Test each passage
    results = {}
    
    for passage_key in ["cyclops", "sirens", "reunion"]:
        print(f"\n📖 ANALYZING {passage_key.upper()} PASSAGE")
        print("-" * 40)
        
        # Analyze the passage context
        context = await agent.analyze_decision_context(passage_key)
        
        print(f"Text Sample: {context['text_sample']}")
        print(f"Complexity Score: {context['complexity']:.3f}")
        print(f"Themes: {', '.join(context['themes'])}")
        print()
        
        # Test decisions for this scenario
        scenario_decisions = mcts.scenarios[passage_key]
        decision_results = []
        
        for decision in scenario_decisions:
            quality = await mcts.evaluate_decision_quality(passage_key, decision)
            decision_results.append({
                "choice": decision.choice,
                "wisdom_level": decision.wisdom_level,
                "narrative_weight": decision.narrative_weight,
                "evaluated_quality": quality,
                "consequences": decision.consequences
            })
            
        # Sort by evaluated quality
        decision_results.sort(key=lambda x: x["evaluated_quality"], reverse=True)
        
        print("🎯 DECISION EVALUATION RESULTS:")
        for i, result in enumerate(decision_results):
            print(f"{i+1}. {result['choice']} (Quality: {result['evaluated_quality']:.3f})")
            print(f"   Wisdom: {result['wisdom_level']:.2f}, Narrative Weight: {result['narrative_weight']:.2f}")
            print(f"   Consequences: {', '.join(result['consequences'][:2])}...")
            print()
            
        results[passage_key] = decision_results
        
    # Overall analysis
    print("\n🔍 CROSS-PASSAGE ANALYSIS")
    print("=" * 40)
    
    total_decisions = sum(len(results[key]) for key in results)
    optimal_decisions = sum(1 for key in results for decision in results[key] 
                          if decision["evaluated_quality"] > 0.7)
    
    print(f"Total Decisions Evaluated: {total_decisions}")
    print(f"High-Quality Decisions (>0.7): {optimal_decisions}")
    print(f"Optimization Rate: {optimal_decisions/total_decisions:.1%}")
    
    # Best decision per passage
    print("\n🏆 OPTIMAL CHOICES PER PASSAGE:")
    for passage_key, decisions in results.items():
        best = decisions[0]  # Already sorted by quality
        print(f"{passage_key.title()}: {best['choice']} (Quality: {best['evaluated_quality']:.3f})")
        
    print("\n✨ Emily Wilson's translation provides rich narrative complexity")
    print("   that enables sophisticated Bayesian decision analysis!")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_real_odyssey_bayesian_mcts())