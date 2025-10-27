#!/usr/bin/env python3
"""
🎭 PIERCING SENTIMENT + NARRATIVE CONTEXT INTEGRATION
====================================================
Advanced sentiment analysis pierced by narrative intelligence
Standalone system with embedded Bayesian sophistication
"""

import numpy as np
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import re


class SentimentDepth(Enum):
    """Progressive sentiment analysis complexity levels"""
    SURFACE = "surface"           # Basic positive/negative
    EMOTIONAL = "emotional"       # Multi-dimensional emotions
    ARCHETYPAL = "archetypal"     # Hero's journey emotional patterns
    NARRATIVE = "narrative"       # Full story-context sentiment
    TEMPORAL = "temporal"         # Evolution across narrative time
    COSMIC = "cosmic"            # Universal human experience patterns


class NarrativeArc(Enum):
    """Epic narrative arc stages"""
    INVOCATION = "invocation"
    DEPARTURE = "departure" 
    TRIALS = "trials"
    WANDERING = "wandering"
    RECOGNITION = "recognition"
    RESOLUTION = "resolution"


@dataclass
class EmotionalSignature:
    """Multi-dimensional emotional understanding"""
    primary_emotion: str
    intensity: float
    archetypal_resonance: Dict[str, float]
    narrative_function: str
    temporal_direction: str  # rising, falling, stable, oscillating
    cosmic_theme: str
    confidence: float
    narrative_enhancement: float


@dataclass
class PiercingSentimentResult:
    """Deep sentiment analysis with narrative context"""
    text: str
    surface_sentiment: Dict[str, float]
    emotional_signature: EmotionalSignature
    narrative_alignment: float
    archetypal_patterns: Dict[str, float]
    temporal_sentiment_evolution: List[Tuple[float, float]]  # (time, sentiment)
    contextual_sentiment_shift: float
    narrative_insights: List[str]
    recommendation: str
    piercing_depth: SentimentDepth
    bayesian_confidence: float


class PiercingSentimentNarrative:
    """Advanced sentiment analysis pierced by narrative intelligence"""
    
    def __init__(self):
        # Archetypal emotional patterns from epic training
        self.archetypal_emotions = {
            "hero_call": {"courage": 0.8, "uncertainty": 0.6, "hope": 0.7, "destiny": 0.9},
            "threshold_crossing": {"fear": 0.7, "determination": 0.9, "transformation": 0.8, "courage": 0.6},
            "trials": {"suffering": 0.8, "perseverance": 0.9, "growth": 0.6, "wisdom": 0.7},
            "revelation": {"wisdom": 0.9, "clarity": 0.8, "understanding": 0.9, "truth": 0.8},
            "return": {"fulfillment": 0.8, "integration": 0.7, "completion": 0.9, "peace": 0.8}
        }
        
        # Narrative function emotional mappings
        self.narrative_functions = {
            "inciting_incident": {"tension": 0.9, "anticipation": 0.8, "change": 0.7},
            "rising_action": {"intensity": 0.7, "complexity": 0.8, "building": 0.9},
            "climax": {"peak_emotion": 1.0, "transformation": 0.9, "decisive": 0.8},
            "falling_action": {"resolution": 0.6, "understanding": 0.7, "consequence": 0.5},
            "denouement": {"closure": 0.8, "reflection": 0.7, "peace": 0.9}
        }
        
        # Cosmic themes from universal human experience
        self.cosmic_themes = {
            "mortality": {"acceptance": 0.8, "transcendence": 0.7, "fate": 0.9},
            "love": {"connection": 0.9, "vulnerability": 0.6, "devotion": 0.8},
            "power": {"ambition": 0.7, "corruption": 0.5, "responsibility": 0.8},
            "identity": {"self_discovery": 0.8, "authenticity": 0.9, "recognition": 0.7},
            "redemption": {"forgiveness": 0.8, "renewal": 0.9, "grace": 0.7},
            "sacrifice": {"nobility": 0.9, "loss": 0.7, "meaning": 0.8, "duty": 0.6}
        }
        
        # Bayesian narrative intelligence patterns
        self.narrative_intelligence = {
            "character_indicators": {
                "odysseus": ["clever", "cunning", "wanderer", "hero", "traveler", "king"],
                "athena": ["wise", "wisdom", "owl", "goddess", "divine", "guidance"],
                "penelope": ["faithful", "weaving", "waiting", "loyal", "patience", "home"],
                "telemachus": ["son", "young", "journey", "growth", "learning", "courage"]
            },
            "arc_patterns": {
                NarrativeArc.INVOCATION: ["call", "destiny", "gods", "divine", "begin", "summon"],
                NarrativeArc.DEPARTURE: ["leave", "journey", "start", "threshold", "cross", "venture"],
                NarrativeArc.TRIALS: ["test", "challenge", "trial", "struggle", "endure", "overcome"],
                NarrativeArc.WANDERING: ["lost", "search", "wander", "drift", "explore", "distant"],
                NarrativeArc.RECOGNITION: ["know", "recognize", "reveal", "identity", "truth", "see"],
                NarrativeArc.RESOLUTION: ["home", "return", "end", "complete", "peace", "fulfill"]
            },
            "complexity_indicators": {
                "high": ["paradox", "irony", "metaphor", "symbol", "layered", "profound"],
                "medium": ["choice", "conflict", "emotion", "relationship", "growth", "change"],
                "low": ["simple", "direct", "clear", "obvious", "straightforward", "basic"]
            }
        }
        
    async def pierce_sentiment(self, text: str, depth: SentimentDepth = SentimentDepth.COSMIC) -> PiercingSentimentResult:
        """Pierce through sentiment layers with narrative intelligence"""
        
        # Get base narrative analysis
        narrative_analysis = await self._analyze_narrative_context(text)
        
        # Progressive sentiment piercing
        if depth == SentimentDepth.SURFACE:
            return await self._surface_sentiment(text, narrative_analysis)
        elif depth == SentimentDepth.EMOTIONAL:
            return await self._emotional_sentiment(text, narrative_analysis)
        elif depth == SentimentDepth.ARCHETYPAL:
            return await self._archetypal_sentiment(text, narrative_analysis)
        elif depth == SentimentDepth.NARRATIVE:
            return await self._narrative_sentiment(text, narrative_analysis)
        elif depth == SentimentDepth.TEMPORAL:
            return await self._temporal_sentiment(text, narrative_analysis)
        else:  # COSMIC
            return await self._cosmic_sentiment(text, narrative_analysis)
    
    async def _analyze_narrative_context(self, text: str) -> Dict[str, Any]:
        """Analyze text for narrative intelligence patterns"""
        
        words = text.lower().split()
        
        # Character detection
        characters = []
        for character, indicators in self.narrative_intelligence["character_indicators"].items():
            if any(indicator in text.lower() for indicator in indicators):
                characters.append(character)
        
        # Arc prediction using Bayesian pattern matching
        arc_scores = {}
        for arc, patterns in self.narrative_intelligence["arc_patterns"].items():
            score = sum(1 for word in words if any(pattern in word for pattern in patterns))
            arc_scores[arc] = score / len(words) if words else 0
        
        predicted_arc = max(arc_scores.keys(), key=lambda k: arc_scores[k]) if arc_scores else None
        
        # Complexity assessment
        complexity_scores = {}
        for level, indicators in self.narrative_intelligence["complexity_indicators"].items():
            score = sum(1 for word in words if any(ind in word for ind in indicators))
            complexity_scores[level] = score / len(words) if words else 0
        
        # Narrative sophistication (Bayesian confidence)
        total_complexity = sum(complexity_scores.values())
        if total_complexity > 0:
            sophistication = (complexity_scores["high"] * 0.9 + 
                            complexity_scores["medium"] * 0.6 + 
                            complexity_scores["low"] * 0.3) / total_complexity
        else:
            sophistication = 0.5
        
        # Theme extraction
        themes = []
        theme_words = {
            "wisdom": ["wise", "wisdom", "knowledge", "understanding", "truth"],
            "home": ["home", "family", "return", "homeland", "hearth"],
            "journey": ["journey", "travel", "path", "road", "quest"],
            "identity": ["who", "am", "name", "self", "identity", "true"],
            "divine": ["god", "goddess", "divine", "fate", "destiny", "prayer"]
        }
        
        for theme, keywords in theme_words.items():
            if any(keyword in text.lower() for keyword in keywords):
                themes.append(theme)
        
        return {
            "characters": characters,
            "predicted_arc": predicted_arc,
            "arc_scores": arc_scores,
            "sophistication": sophistication,
            "complexity_scores": complexity_scores,
            "themes": themes,
            "word_count": len(words)
        }
    
    async def _surface_sentiment(self, text: str, narrative: Dict) -> PiercingSentimentResult:
        """Basic sentiment analysis"""
        # Simple positive/negative analysis
        positive_words = ["good", "great", "wonderful", "joy", "love", "hope", "glory", "wisdom", "home"]
        negative_words = ["bad", "terrible", "pain", "sorrow", "fear", "danger", "loss", "death", "suffering"]
        
        words = text.lower().split()
        pos_count = sum(1 for word in words if any(pw in word for pw in positive_words))
        neg_count = sum(1 for word in words if any(nw in word for nw in negative_words))
        
        total = pos_count + neg_count
        if total == 0:
            sentiment = {"positive": 0.5, "negative": 0.5}
        else:
            sentiment = {"positive": pos_count / total, "negative": neg_count / total}
        
        emotional_sig = EmotionalSignature(
            primary_emotion="neutral",
            intensity=0.3,
            archetypal_resonance={},
            narrative_function="exposition",
            temporal_direction="stable",
            cosmic_theme="mundane",
            confidence=0.4
        )
        
        return PiercingSentimentResult(
            text=text,
            surface_sentiment=sentiment,
            emotional_signature=emotional_sig,
            narrative_alignment=0.3,
            archetypal_patterns={},
            temporal_sentiment_evolution=[(0.0, sentiment["positive"])],
            contextual_sentiment_shift=0.0,
            recommendation="Surface-level sentiment detected",
            piercing_depth=SentimentDepth.SURFACE,
            bayesian_confidence=0.4
        )
    
    async def _cosmic_sentiment(self, text: str, narrative: Dict) -> PiercingSentimentResult:
        """Ultimate piercing sentiment analysis with full narrative intelligence"""
        
        # Extract all narrative intelligence features
        characters = narrative.get('characters', [])
        themes = narrative.get('themes', [])
        predicted_arc = narrative.get('predicted_arc')
        complexity = narrative.get('complexity_level')
        sophistication = narrative.get('narrative_sophistication', 0.5)
        
        # Analyze emotional layers
        emotional_analysis = await self._analyze_emotional_layers(text, narrative)
        
        # Archetypal pattern matching
        archetypal_patterns = await self._match_archetypal_patterns(text, predicted_arc, characters)
        
        # Narrative function analysis
        narrative_function = await self._determine_narrative_function(text, sophistication)
        
        # Temporal sentiment evolution
        temporal_evolution = await self._analyze_temporal_sentiment(text, predicted_arc)
        
        # Cosmic theme resonance
        cosmic_theme, cosmic_resonance = await self._identify_cosmic_theme(text, themes, characters)
        
        # Calculate contextual sentiment shift based on narrative intelligence
        base_sentiment = emotional_analysis["base_sentiment"]
        narrative_enhanced_sentiment = await self._apply_narrative_enhancement(
            base_sentiment, archetypal_patterns, narrative_function, cosmic_resonance
        )
        
        contextual_shift = narrative_enhanced_sentiment - base_sentiment
        
        # Create comprehensive emotional signature
        emotional_signature = EmotionalSignature(
            primary_emotion=emotional_analysis["primary_emotion"],
            intensity=emotional_analysis["intensity"],
            archetypal_resonance=archetypal_patterns,
            narrative_function=narrative_function,
            temporal_direction=emotional_analysis["temporal_direction"],
            cosmic_theme=cosmic_theme,
            confidence=sophistication
        )
        
        # Generate sophisticated recommendation
        recommendation = await self._generate_piercing_recommendation(
            emotional_signature, contextual_shift, narrative, complexity
        )
        
        return PiercingSentimentResult(
            text=text,
            surface_sentiment={"positive": base_sentiment, "negative": 1 - base_sentiment},
            emotional_signature=emotional_signature,
            narrative_alignment=sophistication,
            archetypal_patterns=archetypal_patterns,
            temporal_sentiment_evolution=temporal_evolution,
            contextual_sentiment_shift=contextual_shift,
            recommendation=recommendation,
            piercing_depth=SentimentDepth.COSMIC,
            bayesian_confidence=narrative.get('belief_confidence', 0.6)
        )
    
    async def _analyze_emotional_layers(self, text: str, narrative: Dict) -> Dict:
        """Deep emotional analysis using narrative intelligence"""
        
        # Emotional keyword detection enhanced by narrative context
        emotion_words = {
            "joy": ["joy", "happiness", "delight", "glory", "triumph", "celebration"],
            "sorrow": ["sorrow", "grief", "loss", "mourning", "lament", "tears"],
            "fear": ["fear", "terror", "dread", "anxiety", "worry", "danger"],
            "anger": ["anger", "rage", "fury", "wrath", "indignation", "ire"],
            "love": ["love", "affection", "devotion", "tenderness", "care", "home"],
            "wisdom": ["wisdom", "understanding", "insight", "clarity", "truth", "knowledge"],
            "courage": ["courage", "bravery", "valor", "boldness", "determination", "strength"],
            "longing": ["longing", "yearning", "desire", "nostalgia", "homesickness", "missing"]
        }
        
        words = text.lower().split()
        emotion_scores = {}
        
        for emotion, keywords in emotion_words.items():
            score = sum(1 for word in words if any(kw in word for kw in keywords))
            emotion_scores[emotion] = score / len(words) if words else 0
        
        # Find primary emotion
        primary_emotion = max(emotion_scores.keys(), key=lambda k: emotion_scores[k])
        intensity = emotion_scores[primary_emotion]
        
        # Base sentiment calculation
        positive_emotions = ["joy", "love", "wisdom", "courage"]
        negative_emotions = ["sorrow", "fear", "anger"]
        
        pos_score = sum(emotion_scores[e] for e in positive_emotions)
        neg_score = sum(emotion_scores[e] for e in negative_emotions)
        total_emotion = pos_score + neg_score
        
        if total_emotion == 0:
            base_sentiment = 0.5
        else:
            base_sentiment = pos_score / total_emotion
        
        # Temporal direction analysis
        temporal_words = {
            "rising": ["growing", "increasing", "building", "ascending", "rising"],
            "falling": ["fading", "declining", "ending", "descending", "falling"],
            "oscillating": ["between", "sometimes", "alternating", "mixed", "conflicted"],
            "stable": ["steady", "constant", "unchanging", "peaceful", "calm"]
        }
        
        temporal_direction = "stable"
        max_temporal_score = 0
        for direction, keywords in temporal_words.items():
            score = sum(1 for word in words if any(kw in word for kw in keywords))
            if score > max_temporal_score:
                max_temporal_score = score
                temporal_direction = direction
        
        return {
            "primary_emotion": primary_emotion,
            "intensity": intensity,
            "base_sentiment": base_sentiment,
            "temporal_direction": temporal_direction,
            "emotion_scores": emotion_scores
        }
    
    async def _match_archetypal_patterns(self, text: str, predicted_arc: Optional[str], characters: List[str]) -> Dict[str, float]:
        """Match text against archetypal emotional patterns"""
        
        if not predicted_arc:
            return {}
        
        # Map narrative arc to archetypal pattern
        arc_to_archetype = {
            "invocation": "hero_call",
            "departure": "threshold_crossing", 
            "trials": "trials",
            "wandering": "trials",
            "recognition": "revelation",
            "resolution": "return"
        }
        
        archetype = arc_to_archetype.get(predicted_arc, "trials")
        
        if archetype in self.archetypal_emotions:
            base_pattern = self.archetypal_emotions[archetype].copy()
            
            # Enhance based on character presence
            if "odysseus" in [c.lower() for c in characters]:
                for emotion in base_pattern:
                    base_pattern[emotion] *= 1.2  # Amplify for hero presence
            
            return base_pattern
        
        return {}
    
    async def _determine_narrative_function(self, text: str, sophistication: float) -> str:
        """Determine the narrative function of the text"""
        
        function_indicators = {
            "inciting_incident": ["began", "started", "suddenly", "changed", "disrupted"],
            "rising_action": ["then", "next", "building", "growing", "intensifying"],
            "climax": ["finally", "at last", "moment", "peak", "ultimate", "decisive"],
            "falling_action": ["after", "following", "consequence", "result", "resolution"],
            "denouement": ["end", "conclusion", "finally", "peace", "home", "complete"]
        }
        
        words = text.lower().split()
        function_scores = {}
        
        for function, indicators in function_indicators.items():
            score = sum(1 for word in words if any(ind in word for ind in indicators))
            function_scores[function] = score * sophistication  # Weight by narrative sophistication
        
        if not function_scores or max(function_scores.values()) == 0:
            return "exposition"
        
        return max(function_scores.keys(), key=lambda k: function_scores[k])
    
    async def _analyze_temporal_sentiment(self, text: str, predicted_arc: Optional[str]) -> List[Tuple[float, float]]:
        """Analyze how sentiment evolves across narrative time"""
        
        # Split text into temporal segments
        sentences = text.split('.')
        if len(sentences) < 2:
            return [(0.0, 0.5)]
        
        evolution = []
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                # Simple sentiment for each segment
                positive_words = ["good", "joy", "hope", "wisdom", "glory", "love", "home"]
                negative_words = ["pain", "fear", "sorrow", "danger", "loss", "death"]
                
                words = sentence.lower().split()
                pos_count = sum(1 for word in words if any(pw in word for pw in positive_words))
                neg_count = sum(1 for word in words if any(nw in word for nw in negative_words))
                
                total = pos_count + neg_count
                if total == 0:
                    sentiment = 0.5
                else:
                    sentiment = pos_count / total
                
                time_point = i / (len(sentences) - 1)
                evolution.append((time_point, sentiment))
        
        return evolution
    
    async def _identify_cosmic_theme(self, text: str, themes: List[str], characters: List[str]) -> Tuple[str, float]:
        """Identify universal human experience themes"""
        
        # Check for explicit themes
        for theme in themes:
            theme_lower = theme.lower()
            if theme_lower in self.cosmic_themes:
                return theme_lower, 0.8
        
        # Analyze text for cosmic theme indicators
        cosmic_indicators = {
            "mortality": ["death", "mortal", "life", "dying", "fate", "destiny"],
            "love": ["love", "beloved", "heart", "devotion", "family", "home"],
            "power": ["power", "rule", "control", "authority", "command", "strength"],
            "identity": ["who", "am", "self", "name", "true", "identity", "recognition"],
            "redemption": ["forgive", "redeem", "atone", "salvation", "renewal", "grace"],
            "sacrifice": ["sacrifice", "give", "loss", "offer", "noble", "duty"]
        }
        
        words = text.lower().split()
        theme_scores = {}
        
        for theme, indicators in cosmic_indicators.items():
            score = sum(1 for word in words if any(ind in word for ind in indicators))
            theme_scores[theme] = score / len(words) if words else 0
        
        if not theme_scores or max(theme_scores.values()) == 0:
            return "identity", 0.3
        
        best_theme = max(theme_scores.keys(), key=lambda k: theme_scores[k])
        resonance = theme_scores[best_theme]
        
        return best_theme, min(resonance * 3, 1.0)  # Amplify but cap at 1.0
    
    async def _apply_narrative_enhancement(self, base_sentiment: float, archetypal: Dict[str, float], 
                                         function: str, cosmic_resonance: float) -> float:
        """Apply narrative intelligence to enhance sentiment understanding"""
        
        enhancement = base_sentiment
        
        # Archetypal enhancement
        if archetypal:
            positive_archetypes = ["courage", "wisdom", "hope", "fulfillment", "completion"]
            negative_archetypes = ["fear", "suffering", "uncertainty"]
            
            pos_arch_score = sum(archetypal.get(arch, 0) for arch in positive_archetypes)
            neg_arch_score = sum(archetypal.get(arch, 0) for arch in negative_archetypes)
            
            arch_adjustment = (pos_arch_score - neg_arch_score) * 0.2
            enhancement += arch_adjustment
        
        # Narrative function enhancement
        function_adjustments = {
            "inciting_incident": -0.1,  # Usually creates tension
            "rising_action": -0.05,     # Building tension
            "climax": 0.1,              # Peak emotion (positive for resolution)
            "falling_action": 0.05,     # Moving toward resolution
            "denouement": 0.15          # Resolution and closure
        }
        
        if function in function_adjustments:
            enhancement += function_adjustments[function]
        
        # Cosmic resonance enhancement
        enhancement += cosmic_resonance * 0.1
        
        return max(0.0, min(1.0, enhancement))  # Clamp to [0, 1]
    
    async def _generate_piercing_recommendation(self, signature: EmotionalSignature, 
                                              shift: float, narrative: Dict, complexity: str) -> str:
        """Generate sophisticated interpretation recommendation"""
        
        recommendations = []
        
        # Primary emotion analysis
        if signature.intensity > 0.7:
            recommendations.append(f"Strong {signature.primary_emotion} detected - deep emotional engagement")
        elif signature.intensity > 0.4:
            recommendations.append(f"Moderate {signature.primary_emotion} - balanced emotional state")
        else:
            recommendations.append("Subtle emotional undertones - requires careful interpretation")
        
        # Contextual shift analysis
        if abs(shift) > 0.3:
            direction = "elevated" if shift > 0 else "subdued"
            recommendations.append(f"Narrative context significantly {direction} sentiment (+{shift:.3f})")
        elif abs(shift) > 0.1:
            direction = "enhanced" if shift > 0 else "tempered"
            recommendations.append(f"Narrative intelligence {direction} emotional reading")
        
        # Archetypal pattern insights
        if signature.archetypal_resonance:
            dominant_archetype = max(signature.archetypal_resonance.keys(), 
                                   key=lambda k: signature.archetypal_resonance[k])
            recommendations.append(f"Strong {dominant_archetype} archetypal pattern detected")
        
        # Temporal direction guidance
        if signature.temporal_direction != "stable":
            recommendations.append(f"Emotional trajectory: {signature.temporal_direction}")
        
        # Cosmic theme significance
        if signature.cosmic_theme != "mundane":
            recommendations.append(f"Universal theme of {signature.cosmic_theme} resonates deeply")
        
        # Complexity-based interpretation guidance
        if complexity in ["NEURAL_BAYES", "NON_PARAMETRIC"]:
            recommendations.append("High narrative complexity - multiple interpretation layers recommended")
        
        return " • ".join(recommendations)


async def demonstrate_piercing_sentiment():
    """Demonstrate piercing sentiment analysis capabilities"""
    
    print("🎭 PIERCING SENTIMENT + NARRATIVE CONTEXT DEMONSTRATION")
    print("=" * 70)
    print("Advanced Bayesian sentiment enhanced by epic narrative intelligence\n")
    
    piercer = PiercingSentimentNarrative()
    
    # Test texts with varying emotional and narrative complexity
    test_texts = [
        {
            "title": "Hero's Tragic Choice",
            "text": "Odysseus wept as he held his sword above the innocent, knowing that to save his crew, he must become the monster he despised. The gods watched in silence, neither condemning nor blessing his terrible necessity."
        },
        {
            "title": "Divine Comedy", 
            "text": "Athena laughed, her wisdom sparkling like starlight on wine-dark waters. 'Mortal, you think you understand fate, but you are merely dancing to music you cannot hear. Joy comes to those who surrender to the cosmic symphony.'"
        },
        {
            "title": "Homecoming Peace",
            "text": "The old dog raised his head one final time, recognizing his master's voice after twenty years. In that moment of perfect recognition, love transcended time, and both hero and hound found peace."
        },
        {
            "title": "Temporal Anguish",
            "text": "She had waited so long that hope had transformed into something harder—a kind of crystallized faith that cut her every time she touched it. The loom remained empty, threads scattered like broken promises across marble floors."
        }
    ]
    
    for i, test in enumerate(test_texts, 1):
        print(f"🎬 Test {i}/4: {test['title']}")
        print(f"📝 Text: {test['text']}")
        print()
        
        # Demonstrate progressive piercing depths
        for depth in [SentimentDepth.SURFACE, SentimentDepth.COSMIC]:
            result = await piercer.pierce_sentiment(test['text'], depth)
            
            print(f"   🔍 {depth.value.upper()} ANALYSIS:")
            print(f"   🎭 Primary Emotion: {result.emotional_signature.primary_emotion} "
                  f"(intensity: {result.emotional_signature.intensity:.3f})")
            
            if depth == SentimentDepth.COSMIC:
                print(f"   🏛️ Archetypal Patterns: {result.archetypal_patterns}")
                print(f"   📚 Narrative Function: {result.emotional_signature.narrative_function}")
                print(f"   ⏰ Temporal Direction: {result.emotional_signature.temporal_direction}")
                print(f"   🌌 Cosmic Theme: {result.emotional_signature.cosmic_theme}")
                print(f"   📈 Contextual Shift: {result.contextual_sentiment_shift:+.3f}")
                print(f"   🧠 Bayesian Confidence: {result.bayesian_confidence:.3f}")
                print(f"   💡 Recommendation: {result.recommendation}")
            print()
        
        print("-" * 70)
        print()
    
    print("✨ PIERCING COMPLETE!")
    print("🎯 Sentiment + narrative context successfully integrated!")
    print("🧠 Bayesian intelligence enhances emotional understanding!")


if __name__ == "__main__":
    asyncio.run(demonstrate_piercing_sentiment())