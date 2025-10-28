"""
Bayesian Narrative NLP Integration
Combining our Odyssey MCTS system with natural language processing
"""

import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import re
from collections import Counter
import json

from simplified_odyssey_matryoshka import (
    FullOdysseyProcessor, SimplifiedOdysseyMCTS, MatryoshkaLevel
)
from full_odyssey_structure import FullOdysseyStructure, NarrativeDecision, DecisionType
from temporal_bayesian_evolution import TemporalBayesianEvolution

@dataclass
class NLPNarrativeFeatures:
    """NLP features extracted from narrative text"""
    sentiment_scores: Dict[str, float]
    character_mentions: Dict[str, int]
    theme_keywords: List[str]
    temporal_markers: List[str]
    dialogue_ratio: float
    narrative_voice: str
    complexity_indicators: Dict[str, float]
    metaphor_density: float
    action_verb_ratio: float

class BayesianNarrativeNLP:
    """Natural Language Processing enhanced with Bayesian narrative intelligence"""
    
    def __init__(self, odyssey_processor: FullOdysseyProcessor):
        self.odyssey_processor = odyssey_processor
        self.narrative_patterns = self._extract_narrative_patterns()
        self.character_voice_models = self._build_character_models()
        
    def _extract_narrative_patterns(self) -> Dict[str, Any]:
        """Extract linguistic patterns from Odyssey analysis"""
        patterns = {
            "decision_language": {},
            "complexity_indicators": {},
            "arc_progressions": {},
            "character_speech_patterns": {}
        }
        
        # Analyze decision language patterns from our MCTS results
        for result in self.odyssey_processor.processing_results:
            decision = result["decision"]
            mcts_analysis = result["mcts_analysis"]
            
            # Extract language patterns for different decision types
            decision_type = decision.decision_type.value
            if decision_type not in patterns["decision_language"]:
                patterns["decision_language"][decision_type] = {
                    "keywords": [],
                    "complexity_scores": [],
                    "success_indicators": []
                }
            
            # Analyze the decision description for linguistic markers
            description_words = decision.description.lower().split()
            patterns["decision_language"][decision_type]["keywords"].extend(description_words)
            
            complexity_level = getattr(MatryoshkaLevel, mcts_analysis["complexity_level"]).value
            patterns["decision_language"][decision_type]["complexity_scores"].append(complexity_level)
            
            # Success indicator from canonical vs optimal comparison
            canonical_vs_optimal = result["canonical_vs_optimal"]
            success = canonical_vs_optimal["canonical_ranking"] == 1
            patterns["decision_language"][decision_type]["success_indicators"].append(success)
            
        return patterns
        
    def _build_character_models(self) -> Dict[str, Dict[str, Any]]:
        """Build character voice and decision models from Odyssey analysis"""
        characters = {}
        
        for result in self.odyssey_processor.processing_results:
            decision = result["decision"]
            
            for char_name in decision.affected_characters:
                if char_name not in characters:
                    characters[char_name] = {
                        "decision_involvement": [],
                        "complexity_preferences": [],
                        "wisdom_alignment": [],
                        "narrative_weight": []
                    }
                
                char_model = characters[char_name]
                char_model["decision_involvement"].append(decision.decision_type.value)
                
                complexity_level = getattr(MatryoshkaLevel, 
                    result["mcts_analysis"]["complexity_level"]).value
                char_model["complexity_preferences"].append(complexity_level)
                char_model["wisdom_alignment"].append(decision.wisdom_level)
                char_model["narrative_weight"].append(decision.narrative_weight)
                
        return characters
        
    async def analyze_text_with_narrative_intelligence(self, text: str) -> Dict[str, Any]:
        """Analyze text using our Bayesian narrative intelligence"""
        
        # Extract basic NLP features
        nlp_features = self._extract_nlp_features(text)
        
        # Apply narrative intelligence analysis
        narrative_analysis = await self._apply_narrative_intelligence(text, nlp_features)
        
        # Bayesian belief integration
        belief_integration = await self._integrate_bayesian_beliefs(text, narrative_analysis)
        
        return {
            "nlp_features": nlp_features,
            "narrative_analysis": narrative_analysis,
            "bayesian_beliefs": belief_integration,
            "recommendations": await self._generate_narrative_recommendations(text, narrative_analysis)
        }
        
    def _extract_nlp_features(self, text: str) -> NLPNarrativeFeatures:
        """Extract NLP features from text"""
        words = text.lower().split()
        sentences = re.split(r'[.!?]+', text)
        
        # Sentiment analysis (simplified)
        positive_words = ['good', 'great', 'excellent', 'wonderful', 'joy', 'love', 'hope', 'victory']
        negative_words = ['bad', 'terrible', 'awful', 'pain', 'sorrow', 'fear', 'defeat', 'death']
        
        positive_score = sum(1 for word in words if word in positive_words) / len(words)
        negative_score = sum(1 for word in words if word in negative_words) / len(words)
        
        sentiment_scores = {
            "positive": positive_score,
            "negative": negative_score,
            "neutral": 1.0 - positive_score - negative_score
        }
        
        # Character mentions (from our Odyssey knowledge)
        odyssey_characters = ['odysseus', 'penelope', 'telemachus', 'athena', 'zeus', 'poseidon', 
                             'circe', 'calypso', 'antinous', 'eumaeus', 'eurycleia']
        character_mentions = {}
        for char in odyssey_characters:
            count = sum(1 for word in words if char in word.lower())
            if count > 0:
                character_mentions[char] = count
                
        # Theme keywords
        theme_keywords = []
        odyssey_themes = ['home', 'journey', 'loyalty', 'wisdom', 'justice', 'divine', 'honor', 'family']
        for theme in odyssey_themes:
            if theme in text.lower():
                theme_keywords.append(theme)
                
        # Temporal markers
        temporal_words = ['when', 'then', 'after', 'before', 'now', 'once', 'will', 'would', 'past', 'future']
        temporal_markers = [word for word in words if word in temporal_words]
        
        # Dialogue detection
        dialogue_count = text.count('"') + text.count("'")
        dialogue_ratio = dialogue_count / len(text) if text else 0
        
        # Narrative voice (simplified detection)
        first_person = sum(1 for word in words if word in ['i', 'me', 'my', 'we', 'us'])
        third_person = sum(1 for word in words if word in ['he', 'she', 'they', 'him', 'her'])
        
        if first_person > third_person:
            narrative_voice = "first_person"
        elif third_person > first_person:
            narrative_voice = "third_person"
        else:
            narrative_voice = "mixed"
            
        # Complexity indicators
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        unique_word_ratio = len(set(words)) / len(words) if words else 0
        
        complexity_indicators = {
            "sentence_length": min(avg_sentence_length / 20, 1.0),
            "vocabulary_diversity": unique_word_ratio,
            "temporal_complexity": len(temporal_markers) / len(words) if words else 0
        }
        
        # Metaphor density (simplified)
        metaphor_words = ['like', 'as', 'than', '似', 'resemble']
        metaphor_density = sum(1 for word in words if word in metaphor_words) / len(words) if words else 0
        
        # Action verb ratio
        action_verbs = ['fight', 'sail', 'travel', 'battle', 'struggle', 'overcome', 'defeat', 'escape']
        action_verb_ratio = sum(1 for word in words if word in action_verbs) / len(words) if words else 0
        
        return NLPNarrativeFeatures(
            sentiment_scores=sentiment_scores,
            character_mentions=character_mentions,
            theme_keywords=theme_keywords,
            temporal_markers=temporal_markers,
            dialogue_ratio=dialogue_ratio,
            narrative_voice=narrative_voice,
            complexity_indicators=complexity_indicators,
            metaphor_density=metaphor_density,
            action_verb_ratio=action_verb_ratio
        )
        
    async def _apply_narrative_intelligence(self, text: str, features: NLPNarrativeFeatures) -> Dict[str, Any]:
        """Apply our Odyssey-trained narrative intelligence to the text"""
        
        # Determine likely narrative arc based on content
        arc_indicators = {
            "invocation": ["beginning", "start", "call", "muse", "tell"],
            "departure": ["leave", "journey", "sail", "travel", "go"],
            "trials": ["challenge", "test", "battle", "struggle", "overcome"],
            "wandering": ["lost", "search", "wander", "seek", "find"],
            "recognition": ["know", "recognize", "reveal", "identity", "discover"],
            "resolution": ["end", "home", "peace", "justice", "finish"]
        }
        
        predicted_arc = None
        max_score = 0
        
        text_lower = text.lower()
        for arc, indicators in arc_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            if score > max_score:
                max_score = score
                predicted_arc = arc
                
        # Predict decision complexity using our trained patterns
        complexity_score = 0
        
        # Base complexity from text features
        complexity_score += features.complexity_indicators["sentence_length"] * 2
        complexity_score += features.complexity_indicators["vocabulary_diversity"] * 2
        complexity_score += features.complexity_indicators["temporal_complexity"] * 3
        
        # Character involvement complexity
        complexity_score += len(features.character_mentions) * 0.5
        
        # Theme complexity
        complexity_score += len(features.theme_keywords) * 0.3
        
        # Determine likely Matryoshka level
        if complexity_score <= 1:
            predicted_complexity = MatryoshkaLevel.VARIATIONAL_BAYES
        elif complexity_score <= 2:
            predicted_complexity = MatryoshkaLevel.HIERARCHICAL_BAYES
        elif complexity_score <= 3:
            predicted_complexity = MatryoshkaLevel.GAUSSIAN_PROCESSES
        elif complexity_score <= 4:
            predicted_complexity = MatryoshkaLevel.NEURAL_BAYES
        else:
            predicted_complexity = MatryoshkaLevel.NON_PARAMETRIC
            
        # Predict decision types likely to be involved
        likely_decision_types = []
        if any(word in text_lower for word in ['god', 'divine', 'fate', 'prophecy']):
            likely_decision_types.append("divine")
        if any(word in text_lower for word in ['strategy', 'plan', 'wise', 'clever']):
            likely_decision_types.append("strategic")
        if any(word in text_lower for word in ['identity', 'name', 'who', 'reveal']):
            likely_decision_types.append("identity")
        if any(word in text_lower for word in ['loyalty', 'trust', 'betray', 'faithful']):
            likely_decision_types.append("loyalty")
        if any(word in text_lower for word in ['courage', 'brave', 'hero', 'fight']):
            likely_decision_types.append("heroic")
        if any(word in text_lower for word in ['right', 'wrong', 'justice', 'moral']):
            likely_decision_types.append("moral")
            
        return {
            "predicted_arc": predicted_arc,
            "predicted_complexity": predicted_complexity.name,
            "complexity_score": complexity_score,
            "likely_decision_types": likely_decision_types,
            "narrative_sophistication": complexity_score / 5.0  # Normalized
        }
        
    async def _integrate_bayesian_beliefs(self, text: str, narrative_analysis: Dict) -> Dict[str, Any]:
        """Integrate current Bayesian beliefs with text analysis"""
        
        # Get current beliefs from our Odyssey processor
        current_beliefs = {}
        if hasattr(self.odyssey_processor.bayesian_evolution, 'priors'):
            current_beliefs = {
                name: prior.current_value 
                for name, prior in self.odyssey_processor.bayesian_evolution.priors.items()
            }
            
        # Apply beliefs to text interpretation
        belief_influenced_analysis = {}
        
        # Divine intervention likelihood affects interpretation of divine references
        if 'divine_intervention_likely' in current_beliefs:
            divine_belief = current_beliefs['divine_intervention_likely']
            divine_words = sum(1 for word in ['god', 'divine', 'fate', 'athena', 'zeus'] 
                             if word in text.lower())
            belief_influenced_analysis['divine_significance'] = divine_belief * divine_words
            
        # Wisdom vs force belief affects strategic interpretation
        if 'wisdom_over_force' in current_beliefs:
            wisdom_belief = current_beliefs['wisdom_over_force']
            strategic_words = sum(1 for word in ['wise', 'clever', 'strategy', 'plan'] 
                                if word in text.lower())
            belief_influenced_analysis['strategic_wisdom_score'] = wisdom_belief * strategic_words
            
        # Identity revelation safety affects character analysis
        if 'identity_revelation_safe' in current_beliefs:
            identity_belief = current_beliefs['identity_revelation_safe']
            identity_words = sum(1 for word in ['identity', 'name', 'reveal', 'disguise'] 
                               if word in text.lower())
            belief_influenced_analysis['identity_risk_assessment'] = (1 - identity_belief) * identity_words
            
        return {
            "current_beliefs": current_beliefs,
            "belief_influenced_analysis": belief_influenced_analysis,
            "belief_confidence": np.mean(list(current_beliefs.values())) if current_beliefs else 0.5
        }
        
    async def _generate_narrative_recommendations(self, text: str, analysis: Dict) -> List[str]:
        """Generate narrative recommendations based on analysis"""
        recommendations = []
        
        predicted_complexity = analysis.get("predicted_complexity", "VARIATIONAL_BAYES")
        
        # Complexity-based recommendations
        if predicted_complexity in ["NEURAL_BAYES", "NON_PARAMETRIC"]:
            recommendations.append("This text shows high narrative complexity - consider multiple interpretation layers")
            recommendations.append("Advanced Bayesian reasoning recommended for decision analysis")
        elif predicted_complexity in ["VARIATIONAL_BAYES", "HIERARCHICAL_BAYES"]:
            recommendations.append("Straightforward narrative structure detected - standard analysis sufficient")
            
        # Arc-based recommendations
        predicted_arc = analysis.get("predicted_arc")
        if predicted_arc == "trials":
            recommendations.append("Trial narrative detected - expect strategic and heroic decision points")
        elif predicted_arc == "recognition":
            recommendations.append("Recognition scene indicated - identity decisions likely crucial")
        elif predicted_arc == "resolution":
            recommendations.append("Resolution narrative - divine intervention and moral decisions expected")
            
        # Decision type recommendations
        likely_decisions = analysis.get("likely_decision_types", [])
        if "divine" in likely_decisions:
            recommendations.append("Divine decision context - consider fate vs free will dynamics")
        if "identity" in likely_decisions:
            recommendations.append("Identity decision detected - revelation timing critical")
        if "strategic" in likely_decisions:
            recommendations.append("Strategic decision context - wisdom vs force considerations important")
            
        return recommendations

async def demonstrate_nlp_integration():
    """Demonstrate NLP integration with our Bayesian narrative system"""
    
    print("🧠 BAYESIAN NARRATIVE NLP INTEGRATION DEMO")
    print("=" * 60)
    print("Combining Odyssey intelligence with natural language processing")
    print()
    
    # Initialize our trained Odyssey processor
    processor = FullOdysseyProcessor()
    
    # Quick training on a few decisions to get beliefs
    await processor.process_full_epic(max_decisions=5)
    
    # Initialize NLP system with trained processor
    nlp_system = BayesianNarrativeNLP(processor)
    
    # Test texts of different types
    test_texts = [
        {
            "title": "Hero's Dilemma",
            "text": "Odysseus stood at the crossroads, knowing that revealing his true identity might bring both glory and danger. The wise choice would be patience, but his heart burned for recognition."
        },
        {
            "title": "Divine Intervention",
            "text": "Athena appeared in a dream, her owl eyes gleaming with wisdom. The goddess spoke of fate and choice, warning that the gods watch all mortal decisions with keen interest."
        },
        {
            "title": "Simple Journey",
            "text": "The ship sailed peacefully across the wine-dark sea. The crew worked together, their voices harmonizing as they sang old songs of home and family."
        },
        {
            "title": "Recognition Scene",
            "text": "The old nurse gasped as she saw the scar, her weathered hands trembling. Twenty years had passed, but she knew her master's mark. Should she cry out in joy or maintain the secret?"
        }
    ]
    
    for i, test_case in enumerate(test_texts):
        print(f"📝 Test {i+1}: {test_case['title']}")
        print(f"Text: {test_case['text']}")
        print()
        
        # Analyze with our NLP system
        analysis = await nlp_system.analyze_text_with_narrative_intelligence(test_case['text'])
        
        # Show results
        nlp_features = analysis["nlp_features"]
        narrative_analysis = analysis["narrative_analysis"]
        bayesian_beliefs = analysis["bayesian_beliefs"]
        recommendations = analysis["recommendations"]
        
        print(f"🎭 Sentiment: Positive {nlp_features.sentiment_scores['positive']:.2f}, "
              f"Negative {nlp_features.sentiment_scores['negative']:.2f}")
        
        if nlp_features.character_mentions:
            print(f"👥 Characters: {list(nlp_features.character_mentions.keys())}")
            
        if nlp_features.theme_keywords:
            print(f"🎨 Themes: {nlp_features.theme_keywords}")
            
        print(f"📚 Predicted Arc: {narrative_analysis['predicted_arc']}")
        print(f"🪆 Complexity Level: {narrative_analysis['predicted_complexity']}")
        print(f"🧩 Narrative Sophistication: {narrative_analysis['narrative_sophistication']:.2f}")
        
        if narrative_analysis['likely_decision_types']:
            print(f"⚔️ Likely Decisions: {narrative_analysis['likely_decision_types']}")
            
        print(f"🧠 Belief Confidence: {bayesian_beliefs['belief_confidence']:.2f}")
        
        print("💡 Recommendations:")
        for rec in recommendations[:2]:  # Show top 2
            print(f"  • {rec}")
            
        print("-" * 60)
        print()
        
    print("✨ NLP Integration successfully demonstrated!")
    print("🎯 Bayesian narrative intelligence enhances text understanding!")
    print("🔗 Ready for advanced narrative applications!")

if __name__ == "__main__":
    asyncio.run(demonstrate_nlp_integration())