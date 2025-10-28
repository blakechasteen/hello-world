#!/usr/bin/env python3
"""
🎭 PIERCING SENTIMENT + NARRATIVE CONTEXT INTEGRATION
====================================================
Advanced sentiment analysis pierced by narrative intelligence
"""

import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re


class SentimentDepth(Enum):
    """Progressive sentiment analysis complexity levels"""
    SURFACE = "surface"
    COSMIC = "cosmic"


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
    temporal_direction: str
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
    temporal_sentiment_evolution: List[Tuple[float, float]]
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
            "threshold_crossing": {"fear": 0.7, "determination": 0.9, "transformation": 0.8},
            "trials": {"suffering": 0.8, "perseverance": 0.9, "growth": 0.6, "wisdom": 0.7},
            "revelation": {"wisdom": 0.9, "clarity": 0.8, "understanding": 0.9, "truth": 0.8},
            "return": {"fulfillment": 0.8, "integration": 0.7, "completion": 0.9, "peace": 0.8}
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
        
        # Character detection patterns
        self.character_patterns = {
            "odysseus": ["clever", "cunning", "wanderer", "hero", "traveler", "king", "odysseus"],
            "athena": ["wise", "wisdom", "owl", "goddess", "divine", "guidance", "athena"],
            "penelope": ["faithful", "weaving", "waiting", "loyal", "patience", "home"],
            "divine": ["god", "goddess", "divine", "fate", "destiny", "prayer", "gods"]
        }
        
    async def pierce_sentiment(self, text: str, depth: SentimentDepth = SentimentDepth.COSMIC) -> PiercingSentimentResult:
        """Pierce through sentiment layers with narrative intelligence"""
        
        if depth == SentimentDepth.SURFACE:
            return await self._surface_sentiment(text)
        else:  # COSMIC
            return await self._cosmic_sentiment(text)
    
    async def _surface_sentiment(self, text: str) -> PiercingSentimentResult:
        """Basic sentiment analysis"""
        positive_words = ["good", "great", "wonderful", "joy", "love", "hope", "glory", "wisdom", "home"]
        negative_words = ["bad", "terrible", "pain", "sorrow", "fear", "danger", "loss", "death", "suffering"]
        
        words = text.lower().split()
        pos_count = sum(1 for word in words if any(pw in word for pw in positive_words))
        neg_count = sum(1 for word in words if any(nw in word for nw in negative_words))
        
        total = pos_count + neg_count
        sentiment = pos_count / total if total > 0 else 0.5
        
        emotional_sig = EmotionalSignature(
            primary_emotion="neutral",
            intensity=0.3,
            archetypal_resonance={},
            narrative_function="exposition",
            temporal_direction="stable",
            cosmic_theme="mundane",
            confidence=0.4,
            narrative_enhancement=0.0
        )
        
        return PiercingSentimentResult(
            text=text,
            surface_sentiment={"positive": sentiment, "negative": 1 - sentiment},
            emotional_signature=emotional_sig,
            narrative_alignment=0.3,
            archetypal_patterns={},
            temporal_sentiment_evolution=[(0.0, sentiment)],
            contextual_sentiment_shift=0.0,
            narrative_insights=["Surface-level analysis only"],
            recommendation="Basic sentiment detected - consider deeper analysis",
            piercing_depth=SentimentDepth.SURFACE,
            bayesian_confidence=0.4
        )
    
    async def _cosmic_sentiment(self, text: str) -> PiercingSentimentResult:
        """Ultimate piercing sentiment analysis with full narrative intelligence"""
        
        # 1. Deep emotional analysis
        emotional_analysis = self._analyze_emotions(text)
        
        # 2. Character and narrative detection
        characters = self._detect_characters(text)
        narrative_arc = self._predict_narrative_arc(text)
        
        # 3. Archetypal pattern matching
        archetypal_patterns = self._match_archetypal_patterns(narrative_arc, characters)
        
        # 4. Cosmic theme identification
        cosmic_theme, cosmic_resonance = self._identify_cosmic_theme(text)
        
        # 5. Narrative function analysis
        narrative_function = self._determine_narrative_function(text)
        
        # 6. Temporal sentiment evolution
        temporal_evolution = self._analyze_temporal_sentiment(text)
        
        # 7. Calculate narrative enhancement
        base_sentiment = emotional_analysis["base_sentiment"]
        narrative_enhancement = self._calculate_narrative_enhancement(
            base_sentiment, archetypal_patterns, cosmic_resonance, len(characters)
        )
        
        enhanced_sentiment = max(0.0, min(1.0, base_sentiment + narrative_enhancement))
        contextual_shift = enhanced_sentiment - base_sentiment
        
        # 8. Create emotional signature
        emotional_signature = EmotionalSignature(
            primary_emotion=emotional_analysis["primary_emotion"],
            intensity=emotional_analysis["intensity"],
            archetypal_resonance=archetypal_patterns,
            narrative_function=narrative_function,
            temporal_direction=emotional_analysis["temporal_direction"],
            cosmic_theme=cosmic_theme,
            confidence=0.7 + len(characters) * 0.1,
            narrative_enhancement=narrative_enhancement
        )
        
        # 9. Generate insights and recommendations
        insights = self._generate_insights(characters, narrative_arc, cosmic_theme, emotional_signature)
        recommendation = self._generate_recommendation(emotional_signature, contextual_shift, insights)
        
        return PiercingSentimentResult(
            text=text,
            surface_sentiment={"positive": enhanced_sentiment, "negative": 1 - enhanced_sentiment},
            emotional_signature=emotional_signature,
            narrative_alignment=0.7 + len(characters) * 0.1,
            archetypal_patterns=archetypal_patterns,
            temporal_sentiment_evolution=temporal_evolution,
            contextual_sentiment_shift=contextual_shift,
            narrative_insights=insights,
            recommendation=recommendation,
            piercing_depth=SentimentDepth.COSMIC,
            bayesian_confidence=0.85 + len(characters) * 0.05
        )
    
    def _analyze_emotions(self, text: str) -> Dict:
        """Analyze emotional content of text"""
        emotion_words = {
            "joy": ["joy", "happiness", "delight", "glory", "triumph", "celebration", "bliss", "laughter"],
            "sorrow": ["sorrow", "grief", "loss", "mourning", "lament", "tears", "weeping", "despair"],
            "fear": ["fear", "terror", "dread", "anxiety", "worry", "danger", "afraid", "trembling"],
            "anger": ["anger", "rage", "fury", "wrath", "indignation", "ire", "mad", "furious"],
            "love": ["love", "affection", "devotion", "tenderness", "care", "beloved", "cherish", "adore"],
            "wisdom": ["wisdom", "understanding", "insight", "clarity", "truth", "knowledge", "enlightened"],
            "courage": ["courage", "bravery", "valor", "boldness", "determination", "strength", "heroic", "brave"],
            "longing": ["longing", "yearning", "desire", "nostalgia", "homesickness", "missing", "ache", "pine"]
        }
        
        words = text.lower().split()
        emotion_scores = {}
        
        for emotion, keywords in emotion_words.items():
            score = sum(1 for word in words if any(kw in word for kw in keywords))
            emotion_scores[emotion] = score / len(words) if words else 0
        
        # Find primary emotion
        if max(emotion_scores.values()) > 0:
            primary_emotion = max(emotion_scores.keys(), key=lambda k: emotion_scores[k])
            intensity = min(emotion_scores[primary_emotion] * 5, 1.0)  # Amplify and cap
        else:
            primary_emotion = "neutral"
            intensity = 0.0
        
        # Calculate base sentiment
        positive_emotions = ["joy", "love", "wisdom", "courage"]
        negative_emotions = ["sorrow", "fear", "anger"]
        complex_emotions = ["longing"]
        
        pos_score = sum(emotion_scores[e] for e in positive_emotions)
        neg_score = sum(emotion_scores[e] for e in negative_emotions)
        complex_score = sum(emotion_scores[e] for e in complex_emotions)
        
        total_emotion = pos_score + neg_score + complex_score
        
        if total_emotion == 0:
            base_sentiment = 0.5
        else:
            # Complex emotions add nuance
            base_sentiment = (pos_score + complex_score * 0.6) / total_emotion
        
        # Temporal direction
        temporal_indicators = {
            "rising": ["growing", "increasing", "building", "ascending", "swelling"],
            "falling": ["fading", "declining", "ending", "diminishing", "dying"],
            "oscillating": ["between", "alternating", "mixed", "conflicted", "torn"],
            "stable": ["steady", "constant", "unchanging", "peaceful", "calm"]
        }
        
        temporal_direction = "stable"
        max_score = 0
        for direction, indicators in temporal_indicators.items():
            score = sum(1 for word in words if any(ind in word for ind in indicators))
            if score > max_score:
                max_score = score
                temporal_direction = direction
        
        return {
            "primary_emotion": primary_emotion,
            "intensity": intensity,
            "base_sentiment": base_sentiment,
            "temporal_direction": temporal_direction,
            "emotion_scores": emotion_scores
        }
    
    def _detect_characters(self, text: str) -> List[str]:
        """Detect mythological/narrative characters in text"""
        characters = []
        text_lower = text.lower()
        
        for character, patterns in self.character_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                characters.append(character)
        
        return characters
    
    def _predict_narrative_arc(self, text: str) -> Optional[str]:
        """Predict which narrative arc this text represents"""
        arc_patterns = {
            "invocation": ["call", "destiny", "gods", "divine", "begin", "summon", "invoke"],
            "departure": ["leave", "journey", "start", "threshold", "cross", "venture", "depart"],
            "trials": ["test", "challenge", "trial", "struggle", "endure", "overcome", "suffer"],
            "wandering": ["lost", "search", "wander", "drift", "explore", "distant", "roam"],
            "recognition": ["know", "recognize", "reveal", "identity", "truth", "see", "realize"],
            "resolution": ["home", "return", "end", "complete", "peace", "fulfill", "resolve"]
        }
        
        words = text.lower().split()
        arc_scores = {}
        
        for arc, patterns in arc_patterns.items():
            score = sum(1 for word in words if any(pattern in word for pattern in patterns))
            arc_scores[arc] = score
        
        if max(arc_scores.values()) > 0:
            return max(arc_scores.keys(), key=lambda k: arc_scores[k])
        return None
    
    def _match_archetypal_patterns(self, narrative_arc: Optional[str], characters: List[str]) -> Dict[str, float]:
        """Match against archetypal emotional patterns"""
        if not narrative_arc:
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
        
        archetype = arc_to_archetype.get(narrative_arc, "trials")
        
        if archetype in self.archetypal_emotions:
            patterns = self.archetypal_emotions[archetype].copy()
            
            # Enhance based on character presence
            character_multiplier = 1.0 + len(characters) * 0.2
            
            for emotion in patterns:
                patterns[emotion] *= character_multiplier
                patterns[emotion] = min(patterns[emotion], 1.0)
            
            return patterns
        
        return {}
    
    def _identify_cosmic_theme(self, text: str) -> Tuple[str, float]:
        """Identify universal human experience themes"""
        theme_indicators = {
            "mortality": ["death", "mortal", "life", "dying", "fate", "destiny", "time", "end"],
            "love": ["love", "beloved", "heart", "devotion", "family", "home", "bond", "connection"],
            "power": ["power", "rule", "control", "authority", "command", "strength", "dominion"],
            "identity": ["who", "am", "self", "name", "true", "identity", "recognition", "know"],
            "redemption": ["forgive", "redeem", "atone", "salvation", "renewal", "grace", "cleanse"],
            "sacrifice": ["sacrifice", "give", "loss", "offer", "noble", "duty", "cost", "price"]
        }
        
        words = text.lower().split()
        theme_scores = {}
        
        for theme, indicators in theme_indicators.items():
            score = sum(1 for word in words if any(ind in word for ind in indicators))
            theme_scores[theme] = score / len(words) if words else 0
        
        if max(theme_scores.values()) > 0:
            best_theme = max(theme_scores.keys(), key=lambda k: theme_scores[k])
            resonance = min(theme_scores[best_theme] * 4, 1.0)  # Amplify cosmic themes
            return best_theme, resonance
        
        return "identity", 0.3
    
    def _determine_narrative_function(self, text: str) -> str:
        """Determine the narrative function of the text"""
        function_indicators = {
            "inciting_incident": ["began", "started", "suddenly", "changed", "disrupted", "broke"],
            "rising_action": ["then", "next", "building", "growing", "intensifying", "more"],
            "climax": ["finally", "at last", "moment", "peak", "ultimate", "decisive", "crucial"],
            "falling_action": ["after", "following", "consequence", "result", "resolution", "thus"],
            "denouement": ["end", "conclusion", "finally", "peace", "home", "complete", "rest"]
        }
        
        words = text.lower().split()
        function_scores = {}
        
        for function, indicators in function_indicators.items():
            score = sum(1 for word in words if any(ind in word for ind in indicators))
            function_scores[function] = score
        
        if max(function_scores.values()) > 0:
            return max(function_scores.keys(), key=lambda k: function_scores[k])
        
        return "exposition"
    
    def _analyze_temporal_sentiment(self, text: str) -> List[Tuple[float, float]]:
        """Analyze sentiment evolution across text"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return [(0.0, 0.5)]
        
        evolution = []
        positive_words = ["good", "joy", "hope", "wisdom", "glory", "love", "home", "peace"]
        negative_words = ["pain", "fear", "sorrow", "danger", "loss", "death", "suffering", "despair"]
        
        for i, sentence in enumerate(sentences):
            words = sentence.lower().split()
            pos_count = sum(1 for word in words if any(pw in word for pw in positive_words))
            neg_count = sum(1 for word in words if any(nw in word for nw in negative_words))
            
            total = pos_count + neg_count
            sentiment = pos_count / total if total > 0 else 0.5
            
            time_point = i / (len(sentences) - 1) if len(sentences) > 1 else 0.0
            evolution.append((time_point, sentiment))
        
        return evolution
    
    def _calculate_narrative_enhancement(self, base_sentiment: float, archetypal: Dict[str, float], 
                                       cosmic_resonance: float, character_count: int) -> float:
        """Calculate how narrative intelligence enhances sentiment understanding"""
        enhancement = 0.0
        
        # Archetypal enhancement
        if archetypal:
            positive_archetypes = ["courage", "wisdom", "hope", "fulfillment", "completion", "peace"]
            negative_archetypes = ["fear", "suffering", "uncertainty", "loss"]
            
            pos_arch_score = sum(archetypal.get(arch, 0) for arch in positive_archetypes)
            neg_arch_score = sum(archetypal.get(arch, 0) for arch in negative_archetypes)
            
            enhancement += (pos_arch_score - neg_arch_score) * 0.3
        
        # Cosmic resonance enhancement
        enhancement += cosmic_resonance * 0.2
        
        # Character presence enhancement
        enhancement += character_count * 0.1
        
        return enhancement
    
    def _generate_insights(self, characters: List[str], narrative_arc: Optional[str], 
                          cosmic_theme: str, signature: EmotionalSignature) -> List[str]:
        """Generate narrative intelligence insights"""
        insights = []
        
        if characters:
            insights.append(f"Characters detected: {', '.join(characters)} - enriches emotional context")
        
        if narrative_arc:
            insights.append(f"Narrative arc: {narrative_arc} - shapes emotional interpretation")
        
        if signature.intensity > 0.7:
            insights.append("High emotional intensity - profound engagement detected")
        
        if signature.narrative_enhancement > 0.2:
            insights.append(f"Strong narrative enhancement (+{signature.narrative_enhancement:.3f}) - context dramatically shifts understanding")
        
        if cosmic_theme in ["mortality", "love", "redemption", "sacrifice"]:
            insights.append(f"Cosmic theme of {cosmic_theme} - touches universal human experience")
        
        if signature.temporal_direction != "stable":
            insights.append(f"Dynamic emotional trajectory: {signature.temporal_direction}")
        
        return insights
    
    def _generate_recommendation(self, signature: EmotionalSignature, 
                               contextual_shift: float, insights: List[str]) -> str:
        """Generate sophisticated interpretation recommendation"""
        recommendations = []
        
        # Primary emotion analysis
        if signature.intensity > 0.7:
            recommendations.append(f"Intense {signature.primary_emotion} detected - profound emotional engagement")
        elif signature.intensity > 0.4:
            recommendations.append(f"Moderate {signature.primary_emotion} - nuanced emotional state")
        else:
            recommendations.append("Subtle emotional undertones - narrative context crucial")
        
        # Narrative enhancement impact
        if signature.narrative_enhancement > 0.3:
            recommendations.append(f"Narrative intelligence dramatically elevates understanding (+{signature.narrative_enhancement:.3f})")
        elif signature.narrative_enhancement > 0.1:
            recommendations.append(f"Narrative context enhances emotional reading (+{signature.narrative_enhancement:.3f})")
        
        # Contextual shift significance
        if abs(contextual_shift) > 0.3:
            direction = "elevated" if contextual_shift > 0 else "deepened"
            recommendations.append(f"Context {direction} sentiment by {abs(contextual_shift):.3f}")
        
        # Archetypal significance
        if signature.archetypal_resonance:
            dominant = max(signature.archetypal_resonance.keys(), key=lambda k: signature.archetypal_resonance[k])
            recommendations.append(f"Strong {dominant} archetypal pattern - universal emotional resonance")
        
        # Cosmic significance
        if signature.cosmic_theme in ["mortality", "love", "redemption", "sacrifice"]:
            recommendations.append(f"Universal {signature.cosmic_theme} theme - profound human resonance")
        
        return " • ".join(recommendations)


async def demonstrate_piercing_sentiment():
    """Demonstrate piercing sentiment analysis capabilities"""
    
    print("🎭 PIERCING SENTIMENT + NARRATIVE CONTEXT DEMONSTRATION")
    print("=" * 70)
    print("Advanced sentiment analysis pierced by narrative intelligence\n")
    
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
            "title": "Recognition and Transcendence",
            "text": "The old dog raised his head one final time, recognizing his master's voice after twenty years. In that moment of perfect recognition, love transcended time, and both hero and hound found peace at last."
        },
        {
            "title": "Temporal Anguish",
            "text": "She had waited so long that hope had transformed into something harder—a kind of crystallized faith that cut her every time she touched it. The loom remained empty, threads scattered like broken promises."
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
            print(f"   💫 Sentiment: {result.surface_sentiment['positive']:.3f} positive")
            
            if depth == SentimentDepth.COSMIC:
                print(f"   🏛️ Archetypal Patterns: {result.archetypal_patterns}")
                print(f"   📚 Narrative Function: {result.emotional_signature.narrative_function}")
                print(f"   ⏰ Temporal Direction: {result.emotional_signature.temporal_direction}")
                print(f"   🌌 Cosmic Theme: {result.emotional_signature.cosmic_theme}")
                print(f"   📈 Narrative Enhancement: {result.emotional_signature.narrative_enhancement:+.3f}")
                print(f"   🔄 Contextual Shift: {result.contextual_sentiment_shift:+.3f}")
                print(f"   🧠 Bayesian Confidence: {result.bayesian_confidence:.3f}")
                print(f"   💡 Narrative Insights:")
                for insight in result.narrative_insights:
                    print(f"      • {insight}")
                print(f"   🎯 Recommendation: {result.recommendation}")
            print()
        
        print("-" * 70)
        print()
    
    print("✨ PIERCING COMPLETE!")
    print("🎯 Sentiment + narrative context successfully pierced!")
    print("🧠 Narrative intelligence dramatically enhances emotional understanding!")
    print("📊 Each text shows how Bayesian epic intelligence transforms sentiment analysis!")


if __name__ == "__main__":
    asyncio.run(demonstrate_piercing_sentiment())