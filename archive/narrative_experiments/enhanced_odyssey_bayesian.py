"""
Enhanced Real Odyssey Analysis - Deep Bayesian Integration
Using Emily Wilson's translation with sophisticated text analysis
"""

import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import re
from collections import Counter
import hashlib

from emily_wilson_odyssey_excerpts import get_wilson_passages, analyze_narrative_complexity

class DeepTextAnalyzer:
    """Advanced analysis of Emily Wilson's translation"""
    
    def __init__(self):
        self.wilson_passages = get_wilson_passages()
        
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract sophisticated linguistic features from the text"""
        words = text.lower().split()
        sentences = re.split(r'[.!?]+', text)
        
        features = {}
        
        # Lexical complexity
        unique_words = len(set(words))
        features['lexical_diversity'] = unique_words / len(words) if words else 0
        
        # Sentence complexity
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        features['syntactic_complexity'] = min(avg_sentence_length / 20, 1.0)
        
        # Emotional intensity
        emotion_words = {
            'positive': ['glory', 'famous', 'great', 'welcome', 'glad', 'joy', 'love', 'dear'],
            'negative': ['pain', 'suffering', 'wrecked', 'dreadful', 'death', 'lost', 'failed'],
            'intense': ['wept', 'embraced', 'pounded', 'storms', 'destroyed', 'blinded']
        }
        
        for category, word_list in emotion_words.items():
            count = sum(1 for word in words if any(ew in word for ew in word_list))
            features[f'{category}_emotion'] = count / len(words) if words else 0
            
        # Heroic/Epic language markers
        epic_markers = ['cyclops', 'odysseus', 'goddess', 'muse', 'prophecy', 'fate', 'divine']
        features['epic_language'] = sum(1 for word in words if any(em in word for em in epic_markers)) / len(words) if words else 0
        
        # Temporal complexity (multiple timeframes)
        temporal_markers = ['once', 'now', 'when', 'after', 'before', 'always', 'will', 'would']
        features['temporal_complexity'] = sum(1 for word in words if word in temporal_markers) / len(words) if words else 0
        
        # Decision language
        decision_markers = ['choose', 'decide', 'must', 'should', 'will', 'would', 'if', 'but']
        features['decision_language'] = sum(1 for word in words if word in decision_markers) / len(words) if words else 0
        
        return features
        
    def analyze_narrative_arc(self, passage_key: str) -> Dict[str, float]:
        """Analyze where this passage fits in the heroic journey"""
        passage_data = self.wilson_passages[passage_key]
        text = passage_data["text"].lower()
        
        arc_stages = {
            'call_to_adventure': ['journey', 'wander', 'sail', 'go'],
            'trials': ['pain', 'suffer', 'storm', 'battle', 'fight', 'danger'],
            'revelation': ['tell', 'say', 'know', 'learn', 'see', 'understand'],
            'transformation': ['change', 'become', 'grow', 'wise', 'different'],
            'return': ['home', 'ithaca', 'return', 'back', 'welcome']
        }
        
        arc_scores = {}
        for stage, markers in arc_stages.items():
            score = sum(1 for marker in markers if marker in text) / len(markers)
            arc_scores[stage] = score
            
        return arc_scores

class BayesianOdysseyEvaluator:
    """Bayesian evaluation of decisions in Odyssey context"""
    
    def __init__(self):
        self.text_analyzer = DeepTextAnalyzer()
        
    def calculate_prior_beliefs(self, passage_key: str) -> Dict[str, float]:
        """Calculate prior beliefs about optimal strategies for this passage"""
        linguistic_features = self.text_analyzer.extract_linguistic_features(
            self.text_analyzer.wilson_passages[passage_key]["text"]
        )
        arc_analysis = self.text_analyzer.analyze_narrative_arc(passage_key)
        
        # Prior beliefs based on narrative context
        priors = {}
        
        # Divine context favors cautious, respectful approaches
        divine_context = linguistic_features.get('epic_language', 0)
        priors['caution_value'] = 0.3 + (divine_context * 0.4)
        
        # High emotional intensity favors bold action
        emotional_intensity = (linguistic_features.get('intense_emotion', 0) + 
                             linguistic_features.get('negative_emotion', 0))
        priors['boldness_value'] = 0.2 + (emotional_intensity * 0.5)
        
        # Trials context favors strategic thinking
        trial_context = arc_analysis.get('trials', 0)
        priors['strategy_value'] = 0.4 + (trial_context * 0.3)
        
        # Return context favors wisdom and patience
        return_context = arc_analysis.get('return', 0)
        priors['wisdom_value'] = 0.5 + (return_context * 0.3)
        
        return priors
        
    async def bayesian_decision_evaluation(self, passage_key: str, decision_choice: str, 
                                         wisdom_level: float, narrative_weight: float) -> float:
        """Sophisticated Bayesian evaluation of a decision"""
        
        priors = self.calculate_prior_beliefs(passage_key)
        
        # Likelihood calculation based on decision characteristics
        likelihood = 0.5  # Base likelihood
        
        # Adjust likelihood based on priors and decision characteristics
        if 'reveal' in decision_choice or 'immediate' in decision_choice:
            # Bold, revealing decisions
            likelihood *= (priors['boldness_value'] * 1.5 + priors['caution_value'] * 0.5)
        elif 'test' in decision_choice or 'bound' in decision_choice:
            # Strategic, testing decisions
            likelihood *= (priors['strategy_value'] * 1.3 + priors['wisdom_value'] * 1.2)
        elif 'avoid' in decision_choice or 'gradual' in decision_choice:
            # Cautious, patient decisions
            likelihood *= (priors['caution_value'] * 1.4 + priors['wisdom_value'] * 1.1)
            
        # Weight by narrative importance
        likelihood *= (1.0 + narrative_weight * 0.5)
        
        # Wisdom adjustment (higher wisdom should generally be better)
        wisdom_bonus = wisdom_level * 0.3
        
        # Posterior calculation (simplified Bayesian update)
        posterior = likelihood * (0.5 + wisdom_bonus)  # Prior * Likelihood + wisdom
        
        # Add contextual noise
        noise = np.random.normal(0, 0.05)
        final_score = np.clip(posterior + noise, 0.0, 1.0)
        
        return final_score

async def test_enhanced_odyssey_analysis():
    """Test enhanced Bayesian analysis with real Wilson text"""
    print("🏛️  ENHANCED ODYSSEY BAYESIAN ANALYSIS")
    print("=" * 60)
    print("Deep integration with Emily Wilson's translation")
    print()
    
    evaluator = BayesianOdysseyEvaluator()
    
    # Test scenarios with enhanced analysis
    scenarios = {
        "cyclops": [
            ("reveal_true_name", 0.3, 0.9),
            ("stay_anonymous", 0.8, 0.4), 
            ("partial_truth", 0.6, 0.6)
        ],
        "sirens": [
            ("listen_while_bound", 0.7, 0.8),
            ("avoid_completely", 0.6, 0.3),
            ("crew_listens_too", 0.2, 0.5)
        ],
        "reunion": [
            ("test_with_bed_secret", 0.9, 0.9),
            ("immediate_revelation", 0.4, 0.7),
            ("gradual_revelation", 0.7, 0.6)
        ]
    }
    
    all_results = {}
    
    for passage_key, decisions in scenarios.items():
        print(f"\n📖 ENHANCED ANALYSIS: {passage_key.upper()}")
        print("-" * 40)
        
        # Show linguistic analysis
        passage_text = evaluator.text_analyzer.wilson_passages[passage_key]["text"]
        features = evaluator.text_analyzer.extract_linguistic_features(passage_text)
        arc_analysis = evaluator.text_analyzer.analyze_narrative_arc(passage_key)
        priors = evaluator.calculate_prior_beliefs(passage_key)
        
        print(f"Lexical Diversity: {features['lexical_diversity']:.3f}")
        print(f"Epic Language: {features['epic_language']:.3f}")
        print(f"Temporal Complexity: {features['temporal_complexity']:.3f}")
        print(f"Narrative Arc - Trials: {arc_analysis['trials']:.3f}, Return: {arc_analysis['return']:.3f}")
        print(f"Bayesian Priors - Caution: {priors['caution_value']:.3f}, Wisdom: {priors['wisdom_value']:.3f}")
        print()
        
        # Evaluate decisions
        decision_scores = []
        for choice, wisdom, narrative_weight in decisions:
            score = await evaluator.bayesian_decision_evaluation(
                passage_key, choice, wisdom, narrative_weight
            )
            decision_scores.append({
                "choice": choice,
                "wisdom": wisdom,
                "narrative_weight": narrative_weight,
                "bayesian_score": score
            })
            
        # Sort by Bayesian score
        decision_scores.sort(key=lambda x: x["bayesian_score"], reverse=True)
        
        print("🎯 ENHANCED BAYESIAN RANKINGS:")
        for i, result in enumerate(decision_scores):
            print(f"{i+1}. {result['choice']} (Bayesian Score: {result['bayesian_score']:.3f})")
            print(f"   Wisdom: {result['wisdom']:.2f}, Weight: {result['narrative_weight']:.2f}")
            
        all_results[passage_key] = decision_scores
        print()
        
    # Cross-passage wisdom analysis
    print("\n🔍 CROSS-PASSAGE WISDOM PATTERNS")
    print("=" * 40)
    
    high_scoring_decisions = []
    for passage, decisions in all_results.items():
        for decision in decisions:
            if decision["bayesian_score"] > 0.6:
                high_scoring_decisions.append((passage, decision))
                
    print(f"High-scoring decisions (>0.6): {len(high_scoring_decisions)}")
    
    if high_scoring_decisions:
        avg_wisdom = np.mean([d[1]["wisdom"] for d in high_scoring_decisions])
        avg_weight = np.mean([d[1]["narrative_weight"] for d in high_scoring_decisions])
        print(f"Average wisdom level of top decisions: {avg_wisdom:.3f}")
        print(f"Average narrative weight of top decisions: {avg_weight:.3f}")
        
    print("\n🏆 OPTIMAL STRATEGY PER PASSAGE:")
    for passage, decisions in all_results.items():
        best = decisions[0]
        print(f"{passage.title()}: {best['choice']} (Score: {best['bayesian_score']:.3f})")
        
    print("\n✨ Emily Wilson's translation enables sophisticated")
    print("   Bayesian analysis that respects narrative complexity!")
    
    return all_results

if __name__ == "__main__":
    asyncio.run(test_enhanced_odyssey_analysis())