#!/usr/bin/env python3
"""
🪆 NARRATIVE DEPTH PROTOCOL FOR MYTHRL SHUTTLE
==============================================
Integrates Matryoshka progressive depth gating into the Shuttle architecture

This protocol enables the Shuttle to:
1. Assess narrative complexity of queries
2. Extract progressive layers of meaning (Surface → Cosmic)
3. Enhance results with archetypal and mythic insights
4. Provide full provenance of depth progression

Integration with Shuttle:
- LITE: Surface layer only (literal meaning)
- FAST: Surface + Symbolic (metaphorical interpretation)
- FULL: Surface + Symbolic + Archetypal (universal patterns)
- RESEARCH: All 5 layers including Mythic + Cosmic (ultimate meaning)
"""

from typing import Protocol, Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time

# Import from existing modules
import sys
sys.path.insert(0, '.')

from HoloLoom.protocols.types import (
    ComplexityLevel, 
    ProvenceTrace,
    MythRLResult
)
from HoloLoom.narrative_intelligence import (
    NarrativeIntelligence,
    CampbellStage,
    ArchetypeType
)
from HoloLoom.matryoshka_depth import (
    MatryoshkaNarrativeDepth,
    DepthLevel,
    MatryoshkaDepthResult
)


# === NARRATIVE DEPTH PROTOCOL ===

class NarrativeDepthProtocol(Protocol):
    """Protocol for progressive narrative depth analysis."""
    
    async def assess_narrative_complexity(
        self, 
        query: str, 
        context: Dict,
        complexity_level: ComplexityLevel
    ) -> Dict:
        """
        Assess how much narrative depth is needed for this query.
        
        Returns:
            complexity_score: float (0-1)
            depth_needed: DepthLevel
            narrative_indicators: List[str]
        """
        ...
    
    async def extract_depth_layers(
        self, 
        text: str,
        max_depth: DepthLevel,
        trace: ProvenceTrace
    ) -> MatryoshkaDepthResult:
        """
        Extract progressive layers of meaning up to specified depth.
        """
        ...
    
    async def enhance_with_narrative_intelligence(
        self,
        result_data: Dict,
        depth_result: MatryoshkaDepthResult,
        trace: ProvenceTrace
    ) -> Dict:
        """
        Enhance shuttle results with narrative depth insights.
        """
        ...


# === NARRATIVE DEPTH IMPLEMENTATION ===

class NarrativeDepthEngine:
    """
    Production implementation of narrative depth protocol.
    
    Integrates:
    - Matryoshka progressive depth gating
    - Joseph Campbell Hero's Journey analysis
    - Universal character detection
    - Archetypal pattern recognition
    - Mythic truth extraction
    """
    
    def __init__(self):
        self.narrative_intelligence = NarrativeIntelligence()
        self.matryoshka_depth = MatryoshkaNarrativeDepth()
        
        # Mapping shuttle complexity to depth levels
        self.complexity_to_depth = {
            ComplexityLevel.LITE: DepthLevel.SURFACE,
            ComplexityLevel.FAST: DepthLevel.SYMBOLIC,
            ComplexityLevel.FULL: DepthLevel.ARCHETYPAL,
            ComplexityLevel.RESEARCH: DepthLevel.COSMIC
        }
        
        # Narrative indicator patterns
        self.narrative_patterns = {
            'hero_journey': ['journey', 'quest', 'adventure', 'transformation', 'ordeal'],
            'character': ['character', 'hero', 'mentor', 'villain', 'archetype'],
            'meaning': ['meaning', 'truth', 'wisdom', 'lesson', 'insight'],
            'story': ['story', 'narrative', 'plot', 'arc', 'tale'],
            'mythic': ['myth', 'legend', 'epic', 'saga', 'odyssey'],
            'archetypal': ['archetype', 'pattern', 'universal', 'collective', 'symbolic']
        }
    
    async def assess_narrative_complexity(
        self, 
        query: str, 
        context: Dict,
        complexity_level: ComplexityLevel
    ) -> Dict:
        """Assess narrative complexity of query."""
        query_lower = query.lower()
        
        # Count narrative indicators
        narrative_indicators = []
        complexity_score = 0.0
        
        for category, patterns in self.narrative_patterns.items():
            matches = [p for p in patterns if p in query_lower]
            if matches:
                narrative_indicators.append(f"{category}: {', '.join(matches)}")
                complexity_score += len(matches) * 0.15
        
        # Check for named characters (Odysseus, Harry Potter, etc.)
        character_names = ['odysseus', 'athena', 'frodo', 'gandalf', 'harry', 'hamlet', 
                          'arthur', 'merlin', 'zeus', 'odin', 'luke', 'vader']
        character_matches = [name for name in character_names if name in query_lower]
        if character_matches:
            narrative_indicators.append(f"characters: {', '.join(character_matches)}")
            complexity_score += len(character_matches) * 0.2
        
        # Map to depth level (with shuttle complexity as baseline)
        base_depth = self.complexity_to_depth.get(complexity_level, DepthLevel.SURFACE)
        
        # Narrative score can upgrade depth level
        if complexity_score >= 0.8:
            depth_needed = DepthLevel.COSMIC
        elif complexity_score >= 0.5:
            depth_needed = DepthLevel.MYTHIC
        elif complexity_score >= 0.3:
            depth_needed = DepthLevel.ARCHETYPAL
        elif complexity_score >= 0.15:
            depth_needed = DepthLevel.SYMBOLIC
        else:
            depth_needed = base_depth
        
        # Take max of base depth and narrative-indicated depth
        final_depth = max(base_depth, depth_needed, key=lambda d: d.value)
        
        return {
            'complexity_score': complexity_score,
            'depth_needed': final_depth,
            'narrative_indicators': narrative_indicators,
            'character_matches': character_matches,
            'base_depth': base_depth.name,
            'upgraded_depth': final_depth.name if final_depth != base_depth else None
        }
    
    async def extract_depth_layers(
        self, 
        text: str,
        max_depth: DepthLevel,
        trace: ProvenceTrace
    ) -> MatryoshkaDepthResult:
        """Extract progressive layers of meaning."""
        start_time = time.perf_counter()
        
        # Use matryoshka depth analyzer
        result = await self.matryoshka_depth.analyze_depth(text)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Record in provenance
        trace.add_protocol_call(
            'narrative_depth',
            'extract_depth_layers',
            duration_ms,
            f"Max depth: {result.max_depth_achieved.name}, Complexity: {result.total_complexity:.3f}"
        )
        
        return result
    
    async def enhance_with_narrative_intelligence(
        self,
        result_data: Dict,
        depth_result: MatryoshkaDepthResult,
        trace: ProvenceTrace
    ) -> Dict:
        """Enhance results with narrative insights."""
        start_time = time.perf_counter()
        
        # Extract key narrative enhancements
        enhancements = {
            'narrative_depth': {
                'max_depth_achieved': depth_result.max_depth_achieved.name,
                'total_complexity': depth_result.total_complexity,
                'bayesian_confidence': depth_result.bayesian_confidence,
                'gates_unlocked': len([g for g in depth_result.gates_unlocked if g])
            }
        }
        
        # Add deepest meaning
        enhancements['deepest_meaning'] = depth_result.deepest_meaning
        
        # Add symbolic elements if present
        if depth_result.symbolic_layer and depth_result.symbolic_layer.symbolic_elements:
            enhancements['symbolic_elements'] = depth_result.symbolic_layer.symbolic_elements
        
        # Add archetypal resonance if present
        if depth_result.archetypal_layer and depth_result.archetypal_layer.archetypal_resonance:
            top_archetypes = sorted(
                depth_result.archetypal_layer.archetypal_resonance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            enhancements['top_archetypes'] = dict(top_archetypes)
        
        # Add mythic truths if present
        if depth_result.mythic_layer and depth_result.mythic_layer.universal_truths:
            enhancements['mythic_truths'] = depth_result.mythic_layer.universal_truths[:3]
        
        # Add cosmic truth if present
        if depth_result.cosmic_truth:
            enhancements['cosmic_truth'] = depth_result.cosmic_truth
        
        # Add transformation journey
        enhancements['transformation_journey'] = depth_result.transformation_journey
        
        # Merge with existing result data
        enhanced_result = {**result_data}
        enhanced_result['narrative_intelligence'] = enhancements
        
        # Update confidence based on narrative insights
        if depth_result.bayesian_confidence > enhanced_result.get('confidence', 0.0):
            enhanced_result['confidence'] = depth_result.bayesian_confidence
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        trace.add_protocol_call(
            'narrative_depth',
            'enhance_with_intelligence',
            duration_ms,
            f"Added {len(enhancements)} narrative enhancements"
        )
        
        return enhanced_result


# === SHUTTLE INTEGRATION ===

class NarrativeEnhancedShuttle:
    """
    Extended Shuttle with integrated narrative depth intelligence.
    
    This is an example of how to extend the base MythRLShuttle
    with the narrative depth protocol.
    """
    
    def __init__(self, base_shuttle):
        self.shuttle = base_shuttle
        self.narrative_depth = NarrativeDepthEngine()
        
        # Register narrative depth as a protocol
        self.shuttle.register_protocol('narrative_depth', self.narrative_depth)
    
    async def weave_with_narrative_depth(
        self, 
        query: str, 
        context: Optional[Dict] = None
    ) -> MythRLResult:
        """
        Enhanced weave operation with narrative depth intelligence.
        
        Pipeline:
        1. Standard shuttle weaving
        2. Narrative complexity assessment
        3. Progressive depth extraction
        4. Result enhancement with narrative insights
        """
        context = context or {}
        
        # Get base shuttle result
        base_result = await self.shuttle.weave(query, context)
        
        # Assess narrative complexity
        narrative_assessment = await self.narrative_depth.assess_narrative_complexity(
            query,
            context,
            base_result.complexity_level
        )
        
        base_result.provenance.add_shuttle_event(
            'narrative_assessment',
            f"Narrative depth needed: {narrative_assessment['depth_needed'].name}",
            narrative_assessment
        )
        
        # Extract depth layers if narrative content detected
        if narrative_assessment['complexity_score'] > 0.1:
            # Combine query and output for depth analysis
            text_to_analyze = f"{query}\n\n{base_result.output}"
            
            depth_result = await self.narrative_depth.extract_depth_layers(
                text_to_analyze,
                narrative_assessment['depth_needed'],
                base_result.provenance
            )
            
            # Enhance result with narrative intelligence
            enhanced_data = await self.narrative_depth.enhance_with_narrative_intelligence(
                {'output': base_result.output, 'confidence': base_result.confidence},
                depth_result,
                base_result.provenance
            )
            
            # Update base result
            base_result.output = enhanced_data['output']
            base_result.confidence = enhanced_data['confidence']
            
            # Add narrative intelligence to result metadata
            if not hasattr(base_result, 'metadata'):
                base_result.metadata = {}
            base_result.metadata['narrative_intelligence'] = enhanced_data.get('narrative_intelligence')
        
        return base_result


# === DEMONSTRATION ===

async def demonstrate_narrative_enhanced_shuttle():
    """Demonstrate narrative depth integration with Shuttle."""
    print("🪆 NARRATIVE DEPTH + MYTHRL SHUTTLE INTEGRATION")
    print("=" * 80)
    print()
    
    # Import the base shuttle implementation
    from dev.protocol_modules_mythrl import MythRLShuttle, DemoMemoryBackend
    
    # Create base shuttle
    base_shuttle = MythRLShuttle()
    
    # Register basic protocols
    base_shuttle.register_protocol('memory_backend', DemoMemoryBackend())
    
    # Create narrative-enhanced shuttle
    enhanced_shuttle = NarrativeEnhancedShuttle(base_shuttle)
    
    # Test queries with varying narrative depth
    test_queries = [
        {
            'query': 'What is the weather today?',
            'expected': 'LITE/SURFACE - Simple factual query'
        },
        {
            'query': 'Tell me about the hero\'s journey in Star Wars',
            'expected': 'FULL/ARCHETYPAL - Hero\'s journey analysis'
        },
        {
            'query': 'Analyze the archetypal patterns in Odysseus meeting Athena at the crossroads',
            'expected': 'RESEARCH/COSMIC - Deep mythic analysis'
        },
        {
            'query': 'What is the deeper meaning behind Frodo casting the Ring into Mount Doom?',
            'expected': 'RESEARCH/COSMIC - Ultimate meaning extraction'
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"🎬 Test {i}/4: {test['expected']}")
        print(f"{'=' * 80}")
        print(f"Query: {test['query']}")
        print()
        
        result = await enhanced_shuttle.weave_with_narrative_depth(test['query'])
        
        print(f"📊 SHUTTLE RESULT:")
        print(f"   Complexity Level: {result.complexity_level.name}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Protocol Calls: {len(result.provenance.protocol_calls)}")
        print(f"   Shuttle Events: {len(result.provenance.shuttle_events)}")
        print()
        
        if hasattr(result, 'metadata') and result.metadata.get('narrative_intelligence'):
            ni = result.metadata['narrative_intelligence']
            print(f"🪆 NARRATIVE DEPTH:")
            print(f"   Max Depth: {ni['narrative_depth']['max_depth_achieved']}")
            print(f"   Complexity: {ni['narrative_depth']['total_complexity']:.3f}")
            print(f"   Gates Unlocked: {ni['narrative_depth']['gates_unlocked']}/5")
            print()
            
            if 'deepest_meaning' in ni:
                print(f"💎 Deepest Meaning: {ni['deepest_meaning'][:70]}...")
                print()
            
            if 'symbolic_elements' in ni and ni['symbolic_elements']:
                print(f"🔣 Symbolic Elements: {len(ni['symbolic_elements'])} detected")
                for symbol, meaning in list(ni['symbolic_elements'].items())[:2]:
                    print(f"   • {symbol} → {meaning}")
                print()
            
            if 'top_archetypes' in ni:
                print(f"🏛️ Top Archetypes:")
                for archetype, score in ni['top_archetypes'].items():
                    print(f"   • {archetype}: {score:.3f}")
                print()
            
            if 'mythic_truths' in ni:
                print(f"⚡ Mythic Truths:")
                for truth in ni['mythic_truths']:
                    print(f"   • {truth}")
                print()
            
            if 'cosmic_truth' in ni:
                print(f"🌌 Cosmic Truth: {ni['cosmic_truth']}")
                print()
        
        print(f"{'=' * 80}")
        print()
    
    print("✅ NARRATIVE DEPTH INTEGRATION COMPLETE!")
    print("🪆 Shuttle now has full progressive depth intelligence!")
    print("🎯 From simple queries to cosmic truth revelation!")


if __name__ == "__main__":
    asyncio.run(demonstrate_narrative_enhanced_shuttle())
