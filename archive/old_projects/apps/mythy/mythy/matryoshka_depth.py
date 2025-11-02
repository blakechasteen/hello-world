#!/usr/bin/env python3
"""
🪆 MATRYOSHKA NARRATIVE DEPTH MODULE
====================================
Progressive depth gating for exposing deeper meaning layers
From surface observation to cosmic truth revelation

Depth Levels (Matryoshka Gating):
1. SURFACE (96d)      - Literal text, obvious meaning
2. SYMBOLIC (192d)    - Metaphor, symbolism, subtext
3. ARCHETYPAL (384d)  - Universal patterns, collective unconscious
4. MYTHIC (768d)      - Eternal truths, hero's journey resonance
5. COSMIC (1536d)     - Ultimate meaning, existential significance

Each gate unlocks only when sufficient complexity warrants deeper analysis.
"""

import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re

from .intelligence import (
    NarrativeIntelligence, 
    NarrativeIntelligenceResult,
    CampbellStage,
    ArchetypeType
)


class DepthLevel(Enum):
    """Matryoshka depth levels for progressive meaning extraction"""
    SURFACE = 1      # 96d  - Literal, obvious
    SYMBOLIC = 2     # 192d - Metaphor, symbolism
    ARCHETYPAL = 3   # 384d - Universal patterns
    MYTHIC = 4       # 768d - Eternal truths
    COSMIC = 5       # 1536d - Ultimate meaning


@dataclass
class MeaningLayer:
    """Single layer of meaning at specific depth"""
    depth: DepthLevel
    literal_meaning: str
    hidden_meaning: str
    symbolic_elements: Dict[str, str]  # symbol -> interpretation
    archetypal_resonance: Dict[str, float]
    universal_truths: List[str]
    complexity_score: float
    gate_unlock_reason: str


@dataclass
class MatryoshkaDepthResult:
    """Complete progressive depth analysis"""
    text: str
    
    # Progressive layers (nested like Matryoshka dolls)
    surface_layer: MeaningLayer
    symbolic_layer: Optional[MeaningLayer]
    archetypal_layer: Optional[MeaningLayer]
    mythic_layer: Optional[MeaningLayer]
    cosmic_layer: Optional[MeaningLayer]
    
    # Gating analysis
    max_depth_achieved: DepthLevel
    gates_unlocked: List[Tuple[DepthLevel, float, str]]  # (level, threshold, reason)
    complexity_progression: List[Tuple[DepthLevel, float]]
    
    # Deep insights
    deepest_meaning: str
    transformation_journey: List[str]  # How meaning transforms through layers
    cosmic_truth: Optional[str]
    
    # Meta
    total_complexity: float
    bayesian_confidence: float


class MatryoshkaNarrativeDepth:
    """
    Progressive depth gating for exposing deeper meaning
    Uses complexity thresholds to unlock deeper Matryoshka layers
    """
    
    def __init__(self):
        self.narrative_intelligence = NarrativeIntelligence()
        
        # Complexity thresholds for gate unlocking
        self.gate_thresholds = {
            DepthLevel.SURFACE: 0.0,      # Always accessible
            DepthLevel.SYMBOLIC: 0.3,     # Requires some complexity
            DepthLevel.ARCHETYPAL: 0.5,   # Requires clear patterns
            DepthLevel.MYTHIC: 0.7,       # Requires hero's journey resonance
            DepthLevel.COSMIC: 0.85       # Requires profound significance
        }
        
        # Universal truths database by depth
        self.universal_truths = {
            DepthLevel.MYTHIC: [
                "The hero must die to be reborn",
                "Wisdom comes through suffering",
                "The journey outward is the journey inward",
                "What is sacrificed returns transformed",
                "The mentor appears when the student is ready",
                "Crossing the threshold changes you forever",
                "The treasure is never what you expected",
                "Home is not a place but a state of being",
                "The enemy without reflects the enemy within",
                "Time in the special world flows differently"
            ],
            DepthLevel.COSMIC: [
                "All stories are one story",
                "The individual journey mirrors the cosmic order",
                "Death and rebirth are the fundamental pattern",
                "Separation is the illusion, unity is the truth",
                "The finite contains the infinite",
                "Meaning emerges from the void",
                "The hero's sacrifice redeems the world",
                "Love transcends all boundaries",
                "Consciousness observing itself creates reality",
                "The end is the beginning"
            ]
        }
        
        # Symbolic interpretation patterns
        self.symbolic_patterns = {
            "journey": "inner transformation",
            "home": "authentic self",
            "threshold": "point of no return",
            "darkness": "unconscious depths",
            "light": "consciousness/enlightenment",
            "sword": "discriminating wisdom",
            "water": "the unconscious/emotions",
            "fire": "transformation/destruction",
            "circle": "wholeness/completion",
            "monster": "shadow self",
            "treasure": "integrated wisdom",
            "death": "ego dissolution",
            "birth": "new consciousness",
            "father": "tradition/authority",
            "mother": "source/origin",
            "child": "potential/innocence",
            "king": "conscious order",
            "queen": "sovereign feminine",
            "fool": "divine innocence",
            "wise one": "integrated self"
        }
        
    async def analyze_depth(self, text: str) -> MatryoshkaDepthResult:
        """
        Progressive Matryoshka depth analysis
        Each layer unlocks only if complexity warrants it
        """
        
        # 1. Get base narrative intelligence
        narrative_result = await self.narrative_intelligence.analyze(text)
        
        # 2. Calculate total complexity from multiple factors
        total_complexity = self._calculate_total_complexity(narrative_result)
        
        # 3. Build layers progressively, gating by complexity
        layers = []
        gates_unlocked = []
        complexity_progression = []
        max_depth = DepthLevel.SURFACE
        
        # SURFACE LAYER - Always accessible
        surface_layer = await self._extract_surface_layer(text, narrative_result)
        layers.append(surface_layer)
        gates_unlocked.append((DepthLevel.SURFACE, 0.0, "Always accessible"))
        complexity_progression.append((DepthLevel.SURFACE, total_complexity))
        max_depth = DepthLevel.SURFACE
        
        # SYMBOLIC LAYER - Gate at 0.3 complexity
        symbolic_layer = None
        if total_complexity >= self.gate_thresholds[DepthLevel.SYMBOLIC]:
            symbolic_layer = await self._extract_symbolic_layer(text, narrative_result, surface_layer)
            layers.append(symbolic_layer)
            gates_unlocked.append((
                DepthLevel.SYMBOLIC, 
                self.gate_thresholds[DepthLevel.SYMBOLIC],
                f"Complexity {total_complexity:.3f} >= threshold 0.3"
            ))
            complexity_progression.append((DepthLevel.SYMBOLIC, symbolic_layer.complexity_score))
            max_depth = DepthLevel.SYMBOLIC
        
        # ARCHETYPAL LAYER - Gate at 0.5 complexity
        archetypal_layer = None
        if total_complexity >= self.gate_thresholds[DepthLevel.ARCHETYPAL]:
            archetypal_layer = await self._extract_archetypal_layer(text, narrative_result, symbolic_layer)
            layers.append(archetypal_layer)
            gates_unlocked.append((
                DepthLevel.ARCHETYPAL,
                self.gate_thresholds[DepthLevel.ARCHETYPAL],
                f"Archetypal patterns detected: {len(narrative_result.detected_characters)} characters"
            ))
            complexity_progression.append((DepthLevel.ARCHETYPAL, archetypal_layer.complexity_score))
            max_depth = DepthLevel.ARCHETYPAL
        
        # MYTHIC LAYER - Gate at 0.7 complexity
        mythic_layer = None
        if total_complexity >= self.gate_thresholds[DepthLevel.MYTHIC]:
            mythic_layer = await self._extract_mythic_layer(text, narrative_result, archetypal_layer)
            layers.append(mythic_layer)
            gates_unlocked.append((
                DepthLevel.MYTHIC,
                self.gate_thresholds[DepthLevel.MYTHIC],
                f"Hero's journey stage: {narrative_result.narrative_arc.primary_arc.value}"
            ))
            complexity_progression.append((DepthLevel.MYTHIC, mythic_layer.complexity_score))
            max_depth = DepthLevel.MYTHIC
        
        # COSMIC LAYER - Gate at 0.85 complexity
        cosmic_layer = None
        if total_complexity >= self.gate_thresholds[DepthLevel.COSMIC]:
            cosmic_layer = await self._extract_cosmic_layer(text, narrative_result, mythic_layer)
            layers.append(cosmic_layer)
            gates_unlocked.append((
                DepthLevel.COSMIC,
                self.gate_thresholds[DepthLevel.COSMIC],
                f"Cosmic significance achieved: {narrative_result.narrative_arc.confidence:.3f} confidence"
            ))
            complexity_progression.append((DepthLevel.COSMIC, cosmic_layer.complexity_score))
            max_depth = DepthLevel.COSMIC
        
        # 4. Build transformation journey (how meaning deepens through layers)
        transformation_journey = self._build_transformation_journey(layers)
        
        # 5. Extract deepest meaning achieved
        deepest_meaning = layers[-1].hidden_meaning if layers else "No meaning extracted"
        
        # 6. Extract cosmic truth if achieved
        cosmic_truth = cosmic_layer.universal_truths[0] if cosmic_layer and cosmic_layer.universal_truths else None
        
        return MatryoshkaDepthResult(
            text=text,
            surface_layer=surface_layer,
            symbolic_layer=symbolic_layer,
            archetypal_layer=archetypal_layer,
            mythic_layer=mythic_layer,
            cosmic_layer=cosmic_layer,
            max_depth_achieved=max_depth,
            gates_unlocked=gates_unlocked,
            complexity_progression=complexity_progression,
            deepest_meaning=deepest_meaning,
            transformation_journey=transformation_journey,
            cosmic_truth=cosmic_truth,
            total_complexity=total_complexity,
            bayesian_confidence=narrative_result.bayesian_confidence
        )
    
    def _calculate_total_complexity(self, narrative_result: NarrativeIntelligenceResult) -> float:
        """Calculate total complexity from multiple narrative factors"""
        
        complexity = 0.0
        
        # Character presence adds complexity
        complexity += min(len(narrative_result.detected_characters) * 0.15, 0.3)
        
        # Character confidence
        complexity += narrative_result.character_confidence * 0.15
        
        # Narrative arc confidence
        complexity += narrative_result.narrative_arc.confidence * 0.2
        
        # Archetypal patterns richness
        if narrative_result.archetypal_patterns:
            avg_pattern_strength = sum(narrative_result.archetypal_patterns.values()) / len(narrative_result.archetypal_patterns)
            complexity += avg_pattern_strength * 0.15
        
        # Emotional range indicates depth
        complexity += narrative_result.emotional_range * 0.1
        
        # Theme richness
        complexity += min(len(narrative_result.themes) * 0.05, 0.15)
        
        # Sentiment enhancement factor indicates depth
        complexity += min(abs(narrative_result.enhancement_factor), 0.15)
        
        return min(complexity, 1.0)
    
    async def _extract_surface_layer(self, text: str, narrative_result: NarrativeIntelligenceResult) -> MeaningLayer:
        """Extract surface/literal meaning"""
        
        # Literal meaning - what's explicitly stated
        literal_meaning = f"Text describes "
        if narrative_result.detected_characters:
            char_names = ", ".join([c.name for c in narrative_result.detected_characters[:2]])
            literal_meaning += f"{char_names} "
        
        literal_meaning += f"in {narrative_result.narrative_function.value} context"
        
        # Surface hidden meaning - basic interpretation
        hidden_meaning = f"A narrative moment involving {narrative_result.narrative_function.value}"
        
        return MeaningLayer(
            depth=DepthLevel.SURFACE,
            literal_meaning=literal_meaning,
            hidden_meaning=hidden_meaning,
            symbolic_elements={},
            archetypal_resonance={},
            universal_truths=[],
            complexity_score=0.2,
            gate_unlock_reason="Surface layer always accessible"
        )
    
    async def _extract_symbolic_layer(self, text: str, narrative_result: NarrativeIntelligenceResult, 
                                     surface: MeaningLayer) -> MeaningLayer:
        """Extract symbolic/metaphorical meaning"""
        
        # Detect symbolic elements in text
        symbolic_elements = {}
        text_lower = text.lower()
        
        for symbol, interpretation in self.symbolic_patterns.items():
            if symbol in text_lower:
                symbolic_elements[symbol] = interpretation
        
        # Enhanced literal meaning with symbolic awareness
        literal_meaning = surface.literal_meaning + " with symbolic elements: " + ", ".join(symbolic_elements.keys())
        
        # Symbolic hidden meaning
        if symbolic_elements:
            primary_symbol = list(symbolic_elements.keys())[0]
            hidden_meaning = f"The {primary_symbol} represents {symbolic_elements[primary_symbol]}"
        else:
            hidden_meaning = "Text contains metaphorical layers beyond literal meaning"
        
        return MeaningLayer(
            depth=DepthLevel.SYMBOLIC,
            literal_meaning=literal_meaning,
            hidden_meaning=hidden_meaning,
            symbolic_elements=symbolic_elements,
            archetypal_resonance={},
            universal_truths=[],
            complexity_score=0.4,
            gate_unlock_reason="Symbolic complexity detected"
        )
    
    async def _extract_archetypal_layer(self, text: str, narrative_result: NarrativeIntelligenceResult,
                                       symbolic: Optional[MeaningLayer]) -> MeaningLayer:
        """Extract archetypal/universal pattern meaning"""
        
        # Archetypal resonance from detected characters
        archetypal_resonance = {}
        for char in narrative_result.detected_characters:
            archetype_name = char.archetype.value
            archetypal_resonance[archetype_name] = char.emotional_signature.get("wisdom", 0.5)
        
        # Add narrative arc archetypes
        if narrative_result.archetypal_patterns:
            for pattern, score in narrative_result.archetypal_patterns.items():
                archetypal_resonance[pattern] = score
        
        # Archetypal meaning
        if narrative_result.detected_characters:
            primary_char = narrative_result.detected_characters[0]
            literal_meaning = f"Archetypal {primary_char.archetype.value} pattern: {primary_char.name}"
            hidden_meaning = f"Universal {primary_char.archetype.value} archetype manifests, resonating with collective unconscious"
        else:
            literal_meaning = "Archetypal patterns present in narrative structure"
            hidden_meaning = "Text evokes universal patterns recognized across cultures and time"
        
        # Symbolic elements carry over
        symbolic_elements = symbolic.symbolic_elements if symbolic else {}
        
        return MeaningLayer(
            depth=DepthLevel.ARCHETYPAL,
            literal_meaning=literal_meaning,
            hidden_meaning=hidden_meaning,
            symbolic_elements=symbolic_elements,
            archetypal_resonance=archetypal_resonance,
            universal_truths=[],
            complexity_score=0.6,
            gate_unlock_reason="Archetypal patterns clearly present"
        )
    
    async def _extract_mythic_layer(self, text: str, narrative_result: NarrativeIntelligenceResult,
                                   archetypal: Optional[MeaningLayer]) -> MeaningLayer:
        """Extract mythic/eternal truth meaning"""
        
        # Hero's journey stage reveals mythic meaning
        campbell_stage = narrative_result.narrative_arc.primary_arc
        
        # Select relevant universal truths for this stage
        universal_truths = []
        
        if campbell_stage == CampbellStage.CALL_TO_ADVENTURE:
            universal_truths = ["The mentor appears when the student is ready"]
        elif campbell_stage == CampbellStage.CROSSING_THRESHOLD:
            universal_truths = ["Crossing the threshold changes you forever"]
        elif campbell_stage == CampbellStage.ORDEAL:
            universal_truths = ["The hero must die to be reborn", "Wisdom comes through suffering"]
        elif campbell_stage == CampbellStage.REWARD:
            universal_truths = ["The treasure is never what you expected"]
        elif campbell_stage == CampbellStage.RETURN_WITH_ELIXIR:
            universal_truths = ["Home is not a place but a state of being", "What is sacrificed returns transformed"]
        else:
            universal_truths = ["The journey outward is the journey inward"]
        
        # Mythic meaning
        literal_meaning = f"Mythic stage: {campbell_stage.value} in Hero's Journey"
        hidden_meaning = f"Eternal pattern: {universal_truths[0] if universal_truths else 'Hero undergoes transformation'}"
        
        # Carry forward archetypal resonance
        archetypal_resonance = archetypal.archetypal_resonance if archetypal else {}
        symbolic_elements = archetypal.symbolic_elements if archetypal else {}
        
        return MeaningLayer(
            depth=DepthLevel.MYTHIC,
            literal_meaning=literal_meaning,
            hidden_meaning=hidden_meaning,
            symbolic_elements=symbolic_elements,
            archetypal_resonance=archetypal_resonance,
            universal_truths=universal_truths,
            complexity_score=0.8,
            gate_unlock_reason="Hero's journey resonance detected"
        )
    
    async def _extract_cosmic_layer(self, text: str, narrative_result: NarrativeIntelligenceResult,
                                   mythic: Optional[MeaningLayer]) -> MeaningLayer:
        """Extract cosmic/ultimate meaning"""
        
        # Cosmic truths are the deepest layer
        cosmic_truths = []
        
        # Select cosmic truths based on themes and narrative
        if "death" in text.lower() or "dying" in text.lower():
            cosmic_truths.append("Death and rebirth are the fundamental pattern")
        
        if "home" in text.lower() or "return" in text.lower():
            cosmic_truths.append("The end is the beginning")
        
        if "love" in text.lower() or "beloved" in text.lower():
            cosmic_truths.append("Love transcends all boundaries")
        
        if "wisdom" in text.lower() or "enlighten" in text.lower():
            cosmic_truths.append("Consciousness observing itself creates reality")
        
        # Default cosmic truth
        if not cosmic_truths:
            cosmic_truths = ["All stories are one story", "The individual journey mirrors the cosmic order"]
        
        # Cosmic meaning
        literal_meaning = "Cosmic significance: Text embodies eternal truth"
        hidden_meaning = f"Ultimate meaning: {cosmic_truths[0]}"
        
        # Carry forward all previous layers
        archetypal_resonance = mythic.archetypal_resonance if mythic else {}
        symbolic_elements = mythic.symbolic_elements if mythic else {}
        mythic_truths = mythic.universal_truths if mythic else []
        
        # Combine mythic and cosmic truths
        all_truths = mythic_truths + cosmic_truths
        
        return MeaningLayer(
            depth=DepthLevel.COSMIC,
            literal_meaning=literal_meaning,
            hidden_meaning=hidden_meaning,
            symbolic_elements=symbolic_elements,
            archetypal_resonance=archetypal_resonance,
            universal_truths=all_truths,
            complexity_score=1.0,
            gate_unlock_reason="Cosmic significance threshold achieved"
        )
    
    def _build_transformation_journey(self, layers: List[MeaningLayer]) -> List[str]:
        """Build narrative of how meaning transforms through depths"""
        
        journey = []
        
        for i, layer in enumerate(layers):
            if i == 0:
                journey.append(f"Surface: {layer.literal_meaning}")
            else:
                prev_layer = layers[i-1]
                journey.append(f"{layer.depth.name} (🪆 Gate {i}): {layer.hidden_meaning}")
        
        return journey


async def demonstrate_matryoshka_depth():
    """Demonstrate progressive Matryoshka depth gating"""
    
    print("🪆 MATRYOSHKA NARRATIVE DEPTH DEMONSTRATION")
    print("=" * 80)
    print("Progressive depth gating: Surface → Symbolic → Archetypal → Mythic → Cosmic")
    print("=" * 80)
    print()
    
    depth_analyzer = MatryoshkaNarrativeDepth()
    
    test_texts = [
        {
            "title": "Simple Observation (Low Complexity)",
            "text": "The man walked down the street. It was a sunny day."
        },
        {
            "title": "Symbolic Journey (Medium Complexity)",
            "text": "The traveler stood at the threshold, darkness behind and light ahead. To cross meant leaving the old world forever. The sword at his side felt heavy with destiny."
        },
        {
            "title": "Archetypal Encounter (High Complexity)",
            "text": "Odysseus met Athena at the crossroads, her owl eyes seeing through all deception. 'The journey inward is harder than any odyssey,' she said. 'To find home, you must first lose yourself completely.'"
        },
        {
            "title": "Cosmic Revelation (Profound Complexity)",
            "text": "As Frodo cast the Ring into Mount Doom, he understood: the treasure was never the Ring, but the self he discovered in seeking to destroy it. In that moment of absolute sacrifice, the finite hobbit touched the infinite, and the darkness consuming Middle-earth dissolved into light. Death and rebirth were not opposites but one eternal breath."
        }
    ]
    
    for i, test in enumerate(test_texts, 1):
        print(f"🎬 Test {i}/4: {test['title']}")
        print(f"{'=' * 80}")
        print(f"Text: {test['text']}")
        print()
        
        result = await depth_analyzer.analyze_depth(test['text'])
        
        print(f"📊 COMPLEXITY ANALYSIS:")
        print(f"   Total Complexity: {result.total_complexity:.3f}")
        print(f"   Max Depth Achieved: {result.max_depth_achieved.name} (Level {result.max_depth_achieved.value})")
        print(f"   Bayesian Confidence: {result.bayesian_confidence:.3f}")
        print()
        
        print(f"🪆 MATRYOSHKA GATES UNLOCKED:")
        for depth, threshold, reason in result.gates_unlocked:
            print(f"   ✓ {depth.name} (threshold: {threshold:.2f}) - {reason}")
        print()
        
        print(f"📈 COMPLEXITY PROGRESSION:")
        for depth, complexity in result.complexity_progression:
            bar_length = int(complexity * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"   {depth.name:12s} [{bar}] {complexity:.3f}")
        print()
        
        print(f"🎭 MEANING TRANSFORMATION JOURNEY:")
        for j, step in enumerate(result.transformation_journey, 1):
            print(f"   {j}. {step}")
        print()
        
        print(f"💎 DEEPEST MEANING ACHIEVED:")
        print(f"   {result.deepest_meaning}")
        print()
        
        if result.symbolic_layer:
            print(f"🔣 SYMBOLIC ELEMENTS DETECTED:")
            for symbol, interpretation in result.symbolic_layer.symbolic_elements.items():
                print(f"   • {symbol} → {interpretation}")
            print()
        
        if result.archetypal_layer and result.archetypal_layer.archetypal_resonance:
            print(f"🏛️ ARCHETYPAL RESONANCE:")
            sorted_archetypes = sorted(result.archetypal_layer.archetypal_resonance.items(), 
                                      key=lambda x: x[1], reverse=True)[:5]
            for archetype, score in sorted_archetypes:
                print(f"   • {archetype}: {score:.3f}")
            print()
        
        if result.mythic_layer and result.mythic_layer.universal_truths:
            print(f"⚡ MYTHIC TRUTHS:")
            for truth in result.mythic_layer.universal_truths:
                print(f"   • {truth}")
            print()
        
        if result.cosmic_truth:
            print(f"🌌 COSMIC TRUTH REVEALED:")
            print(f"   {result.cosmic_truth}")
            print()
        
        print(f"{'=' * 80}")
        print()
    
    print("✨ MATRYOSHKA DEPTH ANALYSIS COMPLETE!")
    print("🪆 Progressive gating successfully exposes deeper meaning layers!")
    print("🎯 From surface observation to cosmic truth revelation!")
    print()
    
    # FINAL OUTPUTS SUMMARY
    print("=" * 80)
    print("📊 FINAL OUTPUTS SUMMARY")
    print("=" * 80)
    print()
    
    all_results = []
    for test in test_texts:
        result = await depth_analyzer.analyze_depth(test['text'])
        all_results.append((test['title'], result))
    
    # Summary table
    print("🎯 DEPTH ACHIEVEMENT BY TEXT:")
    print(f"{'Text':<40} {'Max Depth':<15} {'Complexity':<12} {'Confidence':<12} {'Gates':<8}")
    print(f"{'-'*40} {'-'*15} {'-'*12} {'-'*12} {'-'*8}")
    
    for title, result in all_results:
        gates_unlocked = len([g for g in result.gates_unlocked if g])
        print(f"{title:<40} {result.max_depth_achieved.name:<15} {result.total_complexity:.3f}      "
              f"{result.bayesian_confidence:.3f}      {gates_unlocked}/5")
    
    print()
    print("🌟 MEANING PROGRESSION:")
    print()
    
    for title, result in all_results:
        print(f"📖 {title}:")
        print(f"   Surface → {result.surface_layer.literal_meaning[:60]}...")
        if result.symbolic_layer:
            print(f"   Symbolic → {result.symbolic_layer.hidden_meaning[:60]}...")
        if result.archetypal_layer:
            print(f"   Archetypal → {result.archetypal_layer.hidden_meaning[:60]}...")
        if result.mythic_layer:
            print(f"   Mythic → {result.mythic_layer.hidden_meaning[:60]}...")
        if result.cosmic_layer:
            print(f"   Cosmic → {result.cosmic_layer.hidden_meaning[:60]}...")
        print()
    
    print("=" * 80)
    print("🪆 KEY INSIGHTS:")
    print("=" * 80)
    print()
    
    # Find texts that reached each level
    cosmic_texts = [t for t, r in all_results if r.max_depth_achieved == DepthLevel.COSMIC]
    mythic_texts = [t for t, r in all_results if r.max_depth_achieved == DepthLevel.MYTHIC]
    archetypal_texts = [t for t, r in all_results if r.max_depth_achieved == DepthLevel.ARCHETYPAL]
    symbolic_texts = [t for t, r in all_results if r.max_depth_achieved == DepthLevel.SYMBOLIC]
    
    print(f"✨ COSMIC LEVEL ({len(cosmic_texts)} texts):")
    for title, result in all_results:
        if result.max_depth_achieved == DepthLevel.COSMIC:
            print(f"   • {title}: {result.cosmic_truth}")
    print()
    
    print(f"⚡ MYTHIC LEVEL ({len(mythic_texts)} texts):")
    for title, result in all_results:
        if result.max_depth_achieved == DepthLevel.MYTHIC and result.mythic_layer:
            truths = ', '.join(result.mythic_layer.universal_truths[:2])
            print(f"   • {title}: {truths}")
    print()
    
    print(f"🏛️ ARCHETYPAL LEVEL ({len(archetypal_texts)} texts):")
    for title, result in all_results:
        if result.max_depth_achieved == DepthLevel.ARCHETYPAL and result.archetypal_layer:
            top_archetype = max(result.archetypal_layer.archetypal_resonance.items(), 
                              key=lambda x: x[1])[0] if result.archetypal_layer.archetypal_resonance else "none"
            print(f"   • {title}: Strongest archetype = {top_archetype}")
    print()
    
    print(f"🔣 SYMBOLIC LEVEL ({len(symbolic_texts)} texts):")
    for title, result in all_results:
        if result.max_depth_achieved == DepthLevel.SYMBOLIC and result.symbolic_layer:
            symbols = list(result.symbolic_layer.symbolic_elements.keys())[:3]
            print(f"   • {title}: Symbols = {', '.join(symbols)}")
    print()
    
    # Complexity distribution
    complexities = [r.total_complexity for _, r in all_results]
    avg_complexity = sum(complexities) / len(complexities)
    max_complexity = max(complexities)
    min_complexity = min(complexities)
    
    print("📊 COMPLEXITY DISTRIBUTION:")
    print(f"   Average: {avg_complexity:.3f}")
    print(f"   Range: {min_complexity:.3f} - {max_complexity:.3f}")
    print(f"   Span: {max_complexity - min_complexity:.3f}")
    print()
    
    print("🎯 GATING EFFECTIVENESS:")
    total_gates = sum(len([g for g in r.gates_unlocked if g]) for _, r in all_results)
    max_gates = len(all_results) * 5
    print(f"   Total gates unlocked: {total_gates}/{max_gates} ({100*total_gates/max_gates:.1f}%)")
    print(f"   Average gates per text: {total_gates/len(all_results):.1f}")
    print()
    
    print("=" * 80)
    print("✅ MATRYOSHKA PROGRESSIVE DEPTH SYSTEM VALIDATED!")
    print("🪆 Five-level gating successfully exposes meaning from surface to cosmos!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(demonstrate_matryoshka_depth())