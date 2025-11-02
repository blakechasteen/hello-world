#!/usr/bin/env python3
"""
Momentum-Based Gating System for HoloLoom Pipeline
==================================================
Dynamic pipeline that builds momentum and gates deeper processing based on query complexity.

Core Insight: Start simple, escalate only when needed.
- Simple queries: Skip complex stages (fast path)
- Complex queries: Build momentum through deeper Matryoshka scales
- WarpSpace: Non-negotiable - but can be lightweight for simple cases

Gating Strategy:
1. Quick assessment → Determine momentum level needed
2. Lightweight processing → Basic embeddings + simple memory
3. Momentum escalation → Deeper Matryoshka scales when confidence is low
4. Full processing → All stages when query demands sophistication

WarpSpace Philosophy: Always present, but complexity gated
- Simple: Basic tensor operations
- Complex: Full mathematical manifold
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

class MomentumLevel(Enum):
    """Momentum levels for progressive processing."""
    INSTANT = "instant"      # <10ms - Cached/trivial responses
    LIGHT = "light"          # <50ms - Basic embeddings + simple routing
    MEDIUM = "medium"        # <150ms - Multi-scale + intelligent routing  
    HEAVY = "heavy"          # <300ms - Full processing + deep analysis
    RESEARCH = "research"    # No limit - Full weaving with all features


@dataclass
class GatingDecision:
    """Decision about what processing level to use."""
    momentum_level: MomentumLevel
    skip_stages: List[str]
    matryoshka_scales: List[int]
    reasoning: str
    confidence_threshold: float


class MomentumGate:
    """
    Intelligent gating system that determines processing depth.
    
    Philosophy: Build momentum only when needed.
    - Start with lightweight processing
    - Escalate when confidence is insufficient
    - Skip expensive stages for simple queries
    """
    
    def __init__(self):
        self.query_patterns = self._build_query_patterns()
        self.momentum_thresholds = {
            MomentumLevel.INSTANT: 0.95,    # Very high confidence - cached/trivial
            MomentumLevel.LIGHT: 0.80,      # High confidence - simple processing
            MomentumLevel.MEDIUM: 0.60,     # Medium confidence - need more analysis
            MomentumLevel.HEAVY: 0.40,      # Low confidence - full processing
            MomentumLevel.RESEARCH: 0.0     # No confidence limit - research mode
        }
    
    def _build_query_patterns(self) -> Dict[str, MomentumLevel]:
        """Build patterns for quick momentum assessment."""
        return {
            # INSTANT patterns (cached/trivial)
            "what is": MomentumLevel.LIGHT,
            "hello": MomentumLevel.INSTANT,
            "help": MomentumLevel.LIGHT,
            "thanks": MomentumLevel.INSTANT,
            
            # LIGHT patterns (simple queries)
            "search for": MomentumLevel.LIGHT,
            "find": MomentumLevel.LIGHT,
            "show me": MomentumLevel.LIGHT,
            "list": MomentumLevel.LIGHT,
            
            # MEDIUM patterns (need analysis)
            "explain": MomentumLevel.MEDIUM,
            "analyze": MomentumLevel.MEDIUM,
            "compare": MomentumLevel.MEDIUM,
            "summarize": MomentumLevel.MEDIUM,
            "how to": MomentumLevel.MEDIUM,
            
            # HEAVY patterns (complex reasoning)
            "design": MomentumLevel.HEAVY,
            "optimize": MomentumLevel.HEAVY,
            "strategy": MomentumLevel.HEAVY,
            "recommend": MomentumLevel.HEAVY,
            "complex": MomentumLevel.HEAVY,
            
            # RESEARCH patterns (no limits)
            "research": MomentumLevel.RESEARCH,
            "experiment": MomentumLevel.RESEARCH,
            "innovate": MomentumLevel.RESEARCH
        }
    
    async def assess_momentum(self, query: str, context: Optional[Dict] = None) -> GatingDecision:
        """
        Assess what momentum level is needed for this query.
        
        Returns gating decision with stages to skip/include.
        """
        query_lower = query.lower()
        
        # 1. Pattern-based quick assessment
        momentum_level = self._pattern_assessment(query_lower)
        
        # 2. Complexity indicators
        complexity_score = self._assess_complexity(query)
        
        # 3. Context requirements
        context_score = self._assess_context_needs(query, context)
        
        # 4. Combine scores and determine final momentum
        final_momentum = self._determine_final_momentum(
            momentum_level, complexity_score, context_score
        )
        
        # 5. Generate gating decision
        decision = self._create_gating_decision(final_momentum, query)
        
        return decision
    
    def _pattern_assessment(self, query_lower: str) -> MomentumLevel:
        """Quick pattern-based assessment."""
        for pattern, level in self.query_patterns.items():
            if pattern in query_lower:
                return level
        
        # Default based on query length and complexity indicators
        if len(query_lower.split()) <= 3:
            return MomentumLevel.LIGHT
        elif len(query_lower.split()) <= 8:
            return MomentumLevel.MEDIUM
        else:
            return MomentumLevel.HEAVY
    
    def _assess_complexity(self, query: str) -> float:
        """Assess query complexity (0.0 = simple, 1.0 = complex)."""
        complexity_indicators = [
            "relationship", "pattern", "strategy", "optimize", "design",
            "analyze", "compare", "synthesize", "integrate", "complex",
            "multi", "cross", "between", "among", "various", "different"
        ]
        
        words = query.lower().split()
        complexity_matches = sum(1 for word in words if any(indicator in word for indicator in complexity_indicators))
        
        # Normalize by query length
        complexity_score = min(1.0, complexity_matches / max(1, len(words) * 0.3))
        
        # Add length penalty for very long queries
        if len(words) > 15:
            complexity_score += 0.2
        
        return min(1.0, complexity_score)
    
    def _assess_context_needs(self, query: str, context: Optional[Dict]) -> float:
        """Assess how much context this query needs."""
        context_indicators = [
            "previous", "earlier", "before", "remember", "context",
            "related", "similar", "like that", "same as", "compared to"
        ]
        
        words = query.lower().split()
        context_matches = sum(1 for word in words if any(indicator in word for indicator in context_indicators))
        
        # High context need if explicit context references
        if context_matches > 0:
            return min(1.0, context_matches / len(words) * 3)
        
        # Medium context need for analytical queries
        analytical_terms = ["explain", "analyze", "why", "how", "what"]
        if any(term in query.lower() for term in analytical_terms):
            return 0.5
        
        return 0.2  # Low context need for simple queries
    
    def _determine_final_momentum(self, pattern_level: MomentumLevel, complexity: float, context_need: float) -> MomentumLevel:
        """Combine assessments to determine final momentum level."""
        # Start with pattern assessment
        base_score = {
            MomentumLevel.INSTANT: 0.0,
            MomentumLevel.LIGHT: 0.2,
            MomentumLevel.MEDIUM: 0.5,
            MomentumLevel.HEAVY: 0.8,
            MomentumLevel.RESEARCH: 1.0
        }[pattern_level]
        
        # Adjust based on complexity and context
        final_score = base_score + (complexity * 0.3) + (context_need * 0.2)
        
        # Map back to momentum level
        if final_score >= 0.9:
            return MomentumLevel.RESEARCH
        elif final_score >= 0.7:
            return MomentumLevel.HEAVY
        elif final_score >= 0.4:
            return MomentumLevel.MEDIUM
        elif final_score >= 0.1:
            return MomentumLevel.LIGHT
        else:
            return MomentumLevel.INSTANT
    
    def _create_gating_decision(self, momentum_level: MomentumLevel, query: str) -> GatingDecision:
        """Create detailed gating decision based on momentum level."""
        
        if momentum_level == MomentumLevel.INSTANT:
            return GatingDecision(
                momentum_level=momentum_level,
                skip_stages=["resonance_shed", "synthesis", "warp_space", "convergence"],
                matryoshka_scales=[],  # Use cached response
                reasoning="Trivial query - use cached or template response",
                confidence_threshold=0.95
            )
        
        elif momentum_level == MomentumLevel.LIGHT:
            return GatingDecision(
                momentum_level=momentum_level,
                skip_stages=["synthesis", "temporal_window"],
                matryoshka_scales=[96],  # Lightweight embeddings only
                reasoning="Simple query - basic processing sufficient",
                confidence_threshold=0.80
            )
        
        elif momentum_level == MomentumLevel.MEDIUM:
            return GatingDecision(
                momentum_level=momentum_level,
                skip_stages=["temporal_window"],  # Skip only temporal complexity
                matryoshka_scales=[96, 192],  # Medium-scale embeddings
                reasoning="Moderate complexity - need multi-scale analysis",
                confidence_threshold=0.60
            )
        
        elif momentum_level == MomentumLevel.HEAVY:
            return GatingDecision(
                momentum_level=momentum_level,
                skip_stages=[],  # Include all stages
                matryoshka_scales=[96, 192, 384],  # Full-scale embeddings
                reasoning="Complex query - full processing required",
                confidence_threshold=0.40
            )
        
        else:  # RESEARCH
            return GatingDecision(
                momentum_level=momentum_level,
                skip_stages=[],  # Include everything + experimental features
                matryoshka_scales=[96, 192, 384, 768],  # Extended scales if available
                reasoning="Research query - maximum capability deployment",
                confidence_threshold=0.0
            )


class MomentumPipeline:
    """
    Pipeline that builds momentum progressively.
    
    Key Philosophy:
    - WarpSpace is NON-NEGOTIABLE (always present)
    - Discrete processing exists behind gating
    - Build momentum only when confidence is insufficient
    """
    
    def __init__(self):
        self.gate = MomentumGate()
        self.confidence_cache = {}  # Cache confidence scores
        
    async def process(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Process query with momentum-based gating.
        
        Returns result with full trace of what stages were executed.
        """
        start_time = time.perf_counter()
        
        # 1. ASSESS MOMENTUM NEEDED
        gating_decision = await self.gate.assess_momentum(query, context)
        
        print(f"🎯 Momentum Level: {gating_decision.momentum_level.value}")
        print(f"📊 Skip Stages: {gating_decision.skip_stages}")
        print(f"🧠 Matryoshka Scales: {gating_decision.matryoshka_scales}")
        print(f"💡 Reasoning: {gating_decision.reasoning}")
        
        # 2. EXECUTE PIPELINE WITH GATING
        pipeline_result = await self._execute_gated_pipeline(query, gating_decision, context)
        
        # 3. CHECK CONFIDENCE - ESCALATE IF NEEDED
        if pipeline_result['confidence'] < gating_decision.confidence_threshold:
            print(f"⚡ ESCALATING: Confidence {pipeline_result['confidence']:.2f} < {gating_decision.confidence_threshold:.2f}")
            escalated_decision = await self._escalate_momentum(gating_decision)
            pipeline_result = await self._execute_gated_pipeline(query, escalated_decision, context)
        
        # 4. FINALIZE RESULT
        duration_ms = (time.perf_counter() - start_time) * 1000
        pipeline_result['duration_ms'] = duration_ms
        pipeline_result['gating_decision'] = gating_decision
        
        print(f"✅ Complete: {duration_ms:.1f}ms, confidence: {pipeline_result['confidence']:.2f}")
        
        return pipeline_result
    
    async def _execute_gated_pipeline(self, query: str, decision: GatingDecision, context: Optional[Dict]) -> Dict:
        """Execute pipeline with specific gating decisions."""
        
        result = {
            'query': query,
            'stages_executed': [],
            'stages_skipped': decision.skip_stages,
            'confidence': 0.0,
            'output': '',
            'warp_space_complexity': 'basic'
        }
        
        # STAGE 1: Pattern Selection (always quick)
        if decision.momentum_level != MomentumLevel.INSTANT:
            pattern_result = await self._execute_pattern_selection(decision)
            result['stages_executed'].append('pattern_selection')
            result['pattern'] = pattern_result
        
        # STAGE 2: Chrono Trigger (conditional)
        if 'temporal_window' not in decision.skip_stages:
            chrono_result = await self._execute_chrono_trigger(query)
            result['stages_executed'].append('chrono_trigger')
            result['temporal_window'] = chrono_result
        
        # STAGE 3: Resonance Shed (gated by Matryoshka scales)
        if 'resonance_shed' not in decision.skip_stages and decision.matryoshka_scales:
            shed_result = await self._execute_resonance_shed(query, decision.matryoshka_scales)
            result['stages_executed'].append('resonance_shed')
            result['features'] = shed_result
            result['confidence'] = shed_result.get('confidence', 0.5)
        
        # STAGE 3.5: Synthesis (conditional)
        if 'synthesis' not in decision.skip_stages:
            synthesis_result = await self._execute_synthesis(result.get('features', {}))
            result['stages_executed'].append('synthesis')
            result['synthesis'] = synthesis_result
            result['confidence'] = max(result['confidence'], synthesis_result.get('confidence', 0.5))
        
        # STAGE 4: WarpSpace (NON-NEGOTIABLE - but complexity gated)
        warp_complexity = self._determine_warp_complexity(decision.momentum_level)
        warp_result = await self._execute_warp_space(query, context, warp_complexity)
        result['stages_executed'].append('warp_space')
        result['warp_space'] = warp_result
        result['warp_space_complexity'] = warp_complexity
        
        # STAGE 5: Convergence Engine (gated)
        if 'convergence' not in decision.skip_stages:
            convergence_result = await self._execute_convergence(result)
            result['stages_executed'].append('convergence')
            result['decision'] = convergence_result
            result['confidence'] = max(result['confidence'], convergence_result.get('confidence', 0.5))
        else:
            # Simple routing when convergence is skipped
            result['decision'] = {'tool': 'respond', 'confidence': 0.8}
        
        # STAGE 6: Tool Execution (always)
        tool_result = await self._execute_tool(result['decision']['tool'], query, context)
        result['stages_executed'].append('tool_execution')
        result['output'] = tool_result['output']
        result['confidence'] = max(result['confidence'], tool_result.get('confidence', 0.7))
        
        return result
    
    def _determine_warp_complexity(self, momentum_level: MomentumLevel) -> str:
        """Determine WarpSpace complexity level - NON-NEGOTIABLE but gated."""
        if momentum_level == MomentumLevel.INSTANT:
            return 'minimal'     # Basic tensor operations
        elif momentum_level == MomentumLevel.LIGHT:
            return 'basic'       # Simple manifold operations
        elif momentum_level == MomentumLevel.MEDIUM:
            return 'standard'    # Full manifold with optimization
        elif momentum_level == MomentumLevel.HEAVY:
            return 'advanced'    # Complex mathematical operations
        else:  # RESEARCH
            return 'experimental'  # All mathematical features
    
    async def _escalate_momentum(self, current_decision: GatingDecision) -> GatingDecision:
        """Escalate to higher momentum level when confidence is insufficient."""
        escalation_map = {
            MomentumLevel.INSTANT: MomentumLevel.LIGHT,
            MomentumLevel.LIGHT: MomentumLevel.MEDIUM,
            MomentumLevel.MEDIUM: MomentumLevel.HEAVY,
            MomentumLevel.HEAVY: MomentumLevel.RESEARCH,
            MomentumLevel.RESEARCH: MomentumLevel.RESEARCH  # Already at max
        }
        
        new_level = escalation_map[current_decision.momentum_level]
        return self.gate._create_gating_decision(new_level, "escalated")
    
    # Mock implementations of pipeline stages
    async def _execute_pattern_selection(self, decision: GatingDecision) -> Dict:
        return {'selected_pattern': decision.momentum_level.value, 'scales': decision.matryoshka_scales}
    
    async def _execute_chrono_trigger(self, query: str) -> Dict:
        return {'window_start': 'now-1h', 'window_end': 'now', 'recency_bias': 0.8}
    
    async def _execute_resonance_shed(self, query: str, scales: List[int]) -> Dict:
        # Simulate feature extraction with confidence based on scales used
        confidence = min(0.9, 0.5 + len(scales) * 0.15)  # More scales = higher confidence
        return {
            'embeddings': {scale: f'embedding_{scale}d' for scale in scales},
            'motifs': ['pattern_1', 'pattern_2'] if len(scales) > 1 else ['pattern_1'],
            'confidence': confidence
        }
    
    async def _execute_synthesis(self, features: Dict) -> Dict:
        confidence = features.get('confidence', 0.5) + 0.1  # Synthesis adds confidence
        return {
            'entities': ['entity_1', 'entity_2'],
            'patterns': ['complex_pattern'],
            'confidence': min(0.95, confidence)
        }
    
    async def _execute_warp_space(self, query: str, context: Optional[Dict], complexity: str) -> Dict:
        """WarpSpace - NON-NEGOTIABLE but complexity-gated."""
        return {
            'complexity_level': complexity,
            'tensor_operations': ['basic', 'manifold', 'projection'] if complexity != 'minimal' else ['basic'],
            'continuous_representation': f'tensor_field_{complexity}',
            'threads_tensioned': 5 if complexity == 'experimental' else 3
        }
    
    async def _execute_convergence(self, pipeline_state: Dict) -> Dict:
        confidence = pipeline_state.get('confidence', 0.5) + 0.1
        return {
            'tool': 'respond',
            'strategy': 'thompson_sampling',
            'confidence': min(0.95, confidence)
        }
    
    async def _execute_tool(self, tool: str, query: str, context: Optional[Dict]) -> Dict:
        return {
            'tool': tool,
            'output': f"Response to: {query}",
            'confidence': 0.8
        }


async def demo_momentum_system():
    """Demonstrate momentum-based gating system."""
    
    print("🚀 MOMENTUM-BASED GATING SYSTEM DEMO")
    print("=" * 60)
    print("Core Philosophy:")
    print("• Build momentum only when needed")
    print("• Skip stages based on query complexity")
    print("• WarpSpace is NON-NEGOTIABLE (but complexity-gated)")
    print("• Escalate when confidence is insufficient")
    print()
    
    pipeline = MomentumPipeline()
    
    test_queries = [
        "hello",  # Should be INSTANT
        "search for beekeeping tips",  # Should be LIGHT
        "explain the relationship between hive health and seasonal patterns",  # Should be MEDIUM
        "design an optimization strategy for multi-apiary management considering complex environmental variables",  # Should be HEAVY
        "research novel approaches to bee colony collapse disorder using advanced pattern recognition"  # Should be RESEARCH
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {query}")
        print('='*60)
        
        result = await pipeline.process(query)
        
        print(f"\n📊 RESULT SUMMARY:")
        print(f"  Stages Executed: {result['stages_executed']}")
        print(f"  Stages Skipped: {result['stages_skipped']}")
        print(f"  WarpSpace Complexity: {result['warp_space_complexity']}")
        print(f"  Final Confidence: {result['confidence']:.2f}")
        print(f"  Duration: {result['duration_ms']:.1f}ms")
        print(f"  Output: {result['output'][:60]}...")


if __name__ == "__main__":
    asyncio.run(demo_momentum_system())