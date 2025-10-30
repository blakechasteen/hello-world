#!/usr/bin/env python3
"""
Adaptive Learning Protocols for mythRL
======================================
Protocols that evolve and improve over time through experience.

Core Features:
1. Learning Pattern Selection - Evolves pattern recognition based on success
2. Adaptive Decision Engine - Improves decision making through feedback
3. Memory Evolution - Knowledge graph grows smarter over time
4. Success Pattern Mining - Discovers what works and what doesn't
5. Protocol Performance Analytics - Tracks and optimizes protocol effectiveness

Philosophy: Protocols should learn from every interaction and continuously improve
their performance. The Shuttle coordinates this learning across all protocols.
"""

import asyncio
import time
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import statistics

# Import base types from the new architecture
try:
    from dev.protocol_modules_mythrl import (
        ComplexityLevel, ProvenceTrace, MythRLResult,
        PatternSelectionProtocol, DecisionEngineProtocol, MemoryBackendProtocol
    )
except ImportError:
    # Fallback definitions for standalone operation
    from enum import Enum
    class ComplexityLevel(Enum):
        LITE = "lite"
        FAST = "fast" 
        FULL = "full"
        RESEARCH = "research"
    
    class ProvenceTrace:
        pass
    
    class MythRLResult:
        pass
    
    class PatternSelectionProtocol:
        pass
    
    class DecisionEngineProtocol:
        pass
    
    class MemoryBackendProtocol:
        pass


@dataclass
class LearningEvent:
    """Record of a learning event for protocol improvement."""
    timestamp: float
    protocol_name: str
    operation: str
    input_context: Dict
    output_result: Dict
    success_score: float  # 0.0-1.0
    execution_time_ms: float
    complexity_level: ComplexityLevel
    user_feedback: Optional[float] = None  # Explicit user satisfaction


@dataclass
class ProtocolPerformanceMetrics:
    """Performance metrics for a protocol over time."""
    protocol_name: str
    total_invocations: int = 0
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    complexity_performance: Dict[ComplexityLevel, float] = field(default_factory=dict)
    recent_performance_trend: List[float] = field(default_factory=list)
    learned_patterns: Dict[str, float] = field(default_factory=dict)
    adaptation_count: int = 0


class LearningPatternSelection:
    """
    Adaptive Learning Pattern Selection Protocol
    
    Features:
    - Learns from query-pattern success pairs
    - Evolves new pattern recognition strategies
    - Adapts to user preferences over time
    - Discovers emergent patterns in usage
    """
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.pattern_success_history = defaultdict(list)  # pattern -> success scores
        self.query_pattern_associations = {}  # query features -> best patterns
        self.emergent_patterns = {}  # discovered patterns
        self.adaptation_history = []
        self.performance_metrics = ProtocolPerformanceMetrics("pattern_selection")
        
        # Learning parameters
        self.min_samples_for_learning = 5
        self.pattern_confidence_threshold = 0.7
        self.emergence_detection_window = 50
    
    async def select_pattern(self, query: str, context: Dict, complexity: ComplexityLevel) -> Dict:
        """Enhanced pattern selection with learning."""
        start_time = time.perf_counter()
        
        # Extract query features for learning
        query_features = self._extract_query_features(query, context)
        
        # Check learned associations first
        learned_pattern = await self._get_learned_pattern(query_features, complexity)
        
        if learned_pattern:
            pattern_result = {
                'selected_pattern': learned_pattern['pattern'],
                'confidence': learned_pattern['confidence'],
                'reasoning': f'Learned pattern (confidence: {learned_pattern["confidence"]:.2f})',
                'source': 'learned',
                'learning_metadata': {
                    'pattern_success_rate': learned_pattern.get('success_rate', 0.0),
                    'sample_count': learned_pattern.get('sample_count', 0)
                }
            }
        else:
            # Fall back to base pattern selection with complexity scaling
            pattern_result = await self._base_pattern_selection(query, context, complexity)
            pattern_result['source'] = 'base'
        
        # Record this selection for learning
        execution_time = (time.perf_counter() - start_time) * 1000
        self._record_pattern_selection(query_features, pattern_result, execution_time, complexity)
        
        return pattern_result
    
    async def record_pattern_success(self, pattern: str, query_features: Dict, success_score: float, 
                                   user_feedback: Optional[float] = None):
        """Record the success of a pattern selection for learning."""
        
        # Combine objective success with user feedback
        combined_score = success_score
        if user_feedback is not None:
            combined_score = (success_score * 0.7) + (user_feedback * 0.3)
        
        # Update pattern success history
        self.pattern_success_history[pattern].append(combined_score)
        
        # Update query-pattern associations
        query_key = self._hash_query_features(query_features)
        if query_key not in self.query_pattern_associations:
            self.query_pattern_associations[query_key] = []
        
        self.query_pattern_associations[query_key].append({
            'pattern': pattern,
            'success_score': combined_score,
            'timestamp': time.time()
        })
        
        # Trigger learning if we have enough samples
        if len(self.pattern_success_history[pattern]) >= self.min_samples_for_learning:
            await self._update_learned_patterns(pattern)
        
        # Check for emergent patterns
        await self._detect_emergent_patterns()
    
    async def _get_learned_pattern(self, query_features: Dict, complexity: ComplexityLevel) -> Optional[Dict]:
        """Get the best learned pattern for given query features."""
        query_key = self._hash_query_features(query_features)
        
        if query_key not in self.query_pattern_associations:
            return None
        
        # Get recent associations for this query type
        recent_associations = [
            assoc for assoc in self.query_pattern_associations[query_key]
            if time.time() - assoc['timestamp'] < 86400 * 7  # Last week
        ]
        
        if not recent_associations:
            return None
        
        # Find the pattern with best success rate
        pattern_scores = defaultdict(list)
        for assoc in recent_associations:
            pattern_scores[assoc['pattern']].append(assoc['success_score'])
        
        best_pattern = None
        best_score = 0.0
        
        for pattern, scores in pattern_scores.items():
            if len(scores) >= 3:  # Need minimum samples
                avg_score = statistics.mean(scores)
                if avg_score > best_score and avg_score > self.pattern_confidence_threshold:
                    best_pattern = pattern
                    best_score = avg_score
        
        if best_pattern:
            return {
                'pattern': best_pattern,
                'confidence': best_score,
                'success_rate': best_score,
                'sample_count': len(pattern_scores[best_pattern])
            }
        
        return None
    
    async def _update_learned_patterns(self, pattern: str):
        """Update learned patterns based on accumulated data."""
        if pattern not in self.pattern_success_history:
            return
        
        scores = self.pattern_success_history[pattern]
        if len(scores) < self.min_samples_for_learning:
            return
        
        # Calculate pattern performance metrics
        avg_score = statistics.mean(scores)
        recent_scores = scores[-10:]  # Last 10 uses
        recent_avg = statistics.mean(recent_scores)
        trend = recent_avg - avg_score
        
        # Update learned patterns
        self.emergent_patterns[pattern] = {
            'avg_success': avg_score,
            'recent_trend': trend,
            'sample_count': len(scores),
            'last_updated': time.time(),
            'confidence': min(0.95, avg_score + (len(scores) / 100))  # Confidence grows with samples
        }
        
        self.adaptation_history.append({
            'timestamp': time.time(),
            'pattern': pattern,
            'adaptation_type': 'pattern_update',
            'new_confidence': self.emergent_patterns[pattern]['confidence'],
            'sample_count': len(scores)
        })
        
        self.performance_metrics.adaptation_count += 1
    
    async def _detect_emergent_patterns(self):
        """Detect new emergent patterns from usage data."""
        if len(self.adaptation_history) < self.emergence_detection_window:
            return
        
        # Analyze recent patterns for emergence
        recent_patterns = defaultdict(int)
        for event in self.adaptation_history[-self.emergence_detection_window:]:
            if event['adaptation_type'] == 'pattern_update':
                recent_patterns[event['pattern']] += 1
        
        # Look for patterns that are being learned frequently
        for pattern, frequency in recent_patterns.items():
            if frequency > 5 and pattern not in self.emergent_patterns:
                # This might be an emergent pattern
                await self._create_emergent_pattern(pattern, frequency)
    
    async def _create_emergent_pattern(self, base_pattern: str, frequency: int):
        """Create a new emergent pattern based on detected usage."""
        emergent_name = f"emergent_{base_pattern}_{int(time.time())}"
        
        self.emergent_patterns[emergent_name] = {
            'base_pattern': base_pattern,
            'emergence_frequency': frequency,
            'created_at': time.time(),
            'confidence': 0.6,  # Start with moderate confidence
            'type': 'emergent'
        }
        
        self.adaptation_history.append({
            'timestamp': time.time(),
            'pattern': emergent_name,
            'adaptation_type': 'pattern_emergence',
            'base_pattern': base_pattern,
            'frequency': frequency
        })
    
    def _extract_query_features(self, query: str, context: Dict) -> Dict:
        """Extract features from query for pattern learning."""
        return {
            'length': len(query.split()),
            'has_question': '?' in query,
            'keywords': [word.lower() for word in query.split()[:5]],  # First 5 words
            'context_size': len(context),
            'time_of_day': time.localtime().tm_hour,
            'contains_analysis': any(word in query.lower() for word in ['analyze', 'explain', 'compare']),
            'contains_creation': any(word in query.lower() for word in ['create', 'generate', 'make']),
            'contains_search': any(word in query.lower() for word in ['find', 'search', 'lookup'])
        }
    
    def _hash_query_features(self, features: Dict) -> str:
        """Create a hash key for query features."""
        # Simple feature hashing based on key characteristics
        key_parts = [
            f"len_{features['length']//3}",  # Bucketed length
            f"q_{features['has_question']}",
            f"ctx_{features['context_size']//5}",  # Bucketed context
            f"analysis_{features['contains_analysis']}",
            f"creation_{features['contains_creation']}",
            f"search_{features['contains_search']}"
        ]
        return "_".join(key_parts)
    
    async def _base_pattern_selection(self, query: str, context: Dict, complexity: ComplexityLevel) -> Dict:
        """Base pattern selection logic."""
        query_lower = query.lower()
        
        # Complexity-aware pattern selection
        if complexity == ComplexityLevel.LITE:
            patterns = ['simple_response', 'direct_answer']
        elif complexity == ComplexityLevel.FAST:
            patterns = ['search_pattern', 'quick_analysis']
        elif complexity == ComplexityLevel.FULL:
            patterns = ['deep_analysis', 'synthesis_pattern']
        else:  # RESEARCH
            patterns = ['research_pattern', 'experimental_approach']
        
        # Select based on query content
        if any(word in query_lower for word in ['analyze', 'explain']):
            selected = patterns[-1] if len(patterns) > 1 else patterns[0]
        elif any(word in query_lower for word in ['search', 'find']):
            selected = patterns[0]
        else:
            selected = patterns[len(patterns)//2] if len(patterns) > 1 else patterns[0]
        
        return {
            'selected_pattern': selected,
            'confidence': 0.7,
            'reasoning': f'Base selection for {complexity.name} complexity'
        }
    
    def _record_pattern_selection(self, query_features: Dict, pattern_result: Dict, 
                                execution_time: float, complexity: ComplexityLevel):
        """Record pattern selection for performance tracking."""
        self.performance_metrics.total_invocations += 1
        
        # Update execution time average
        if self.performance_metrics.avg_execution_time == 0:
            self.performance_metrics.avg_execution_time = execution_time
        else:
            self.performance_metrics.avg_execution_time = (
                (self.performance_metrics.avg_execution_time * 0.9) + 
                (execution_time * 0.1)
            )
        
        # Update complexity performance
        if complexity not in self.performance_metrics.complexity_performance:
            self.performance_metrics.complexity_performance[complexity] = execution_time
        else:
            current = self.performance_metrics.complexity_performance[complexity]
            self.performance_metrics.complexity_performance[complexity] = (
                (current * 0.9) + (execution_time * 0.1)
            )
    
    async def get_learning_summary(self) -> Dict:
        """Get a summary of learning progress."""
        return {
            'total_patterns_learned': len(self.emergent_patterns),
            'total_adaptations': self.performance_metrics.adaptation_count,
            'performance_metrics': {
                'total_invocations': self.performance_metrics.total_invocations,
                'avg_execution_time': self.performance_metrics.avg_execution_time,
                'complexity_performance': {
                    level.name: time_ms for level, time_ms in 
                    self.performance_metrics.complexity_performance.items()
                }
            },
            'emergent_patterns': {
                name: {
                    'confidence': data['confidence'],
                    'type': data.get('type', 'learned'),
                    'sample_count': data.get('sample_count', 0)
                }
                for name, data in self.emergent_patterns.items()
            },
            'recent_adaptations': self.adaptation_history[-10:]  # Last 10 adaptations
        }


class AdaptiveDecisionEngine:
    """
    Adaptive Learning Decision Engine Protocol
    
    Features:
    - Learns from decision outcomes
    - Adapts decision criteria based on success
    - Multi-criteria optimization that improves over time
    - Context-aware decision making
    """
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.decision_history = []
        self.success_patterns = defaultdict(float)
        self.context_decision_mapping = {}
        self.criteria_weights = {'confidence': 0.4, 'speed': 0.3, 'quality': 0.3}
        self.adaptation_count = 0
    
    async def make_decision(self, features: Dict, context: Dict, options: List[Dict]) -> Dict:
        """Enhanced decision making with learning."""
        start_time = time.perf_counter()
        
        # Extract decision context
        decision_context = self._extract_decision_context(features, context)
        
        # Check for learned decision patterns
        learned_decision = await self._get_learned_decision(decision_context, options)
        
        if learned_decision:
            decision = learned_decision
            decision['source'] = 'learned'
        else:
            # Use adaptive criteria weights
            decision = await self._weighted_decision(options, self.criteria_weights)
            decision['source'] = 'adaptive'
        
        # Record decision for learning
        execution_time = (time.perf_counter() - start_time) * 1000
        self._record_decision(decision_context, decision, options, execution_time)
        
        return decision
    
    async def record_decision_outcome(self, decision_id: str, outcome_score: float, 
                                    feedback: Optional[Dict] = None):
        """Record the outcome of a decision for learning."""
        # Find the decision in history
        for decision_record in self.decision_history:
            if decision_record.get('decision_id') == decision_id:
                decision_record['outcome_score'] = outcome_score
                decision_record['feedback'] = feedback
                
                # Update learning
                await self._update_decision_learning(decision_record)
                break
    
    async def _get_learned_decision(self, context: Dict, options: List[Dict]) -> Optional[Dict]:
        """Get learned decision for similar context."""
        context_key = self._hash_decision_context(context)
        
        if context_key in self.context_decision_mapping:
            learned_pattern = self.context_decision_mapping[context_key]
            
            # Find best matching option
            best_option = None
            best_score = 0.0
            
            for option in options:
                similarity = self._calculate_option_similarity(option, learned_pattern['best_option'])
                if similarity > best_score and similarity > 0.7:
                    best_option = option
                    best_score = similarity
            
            if best_option:
                return {
                    'selected_option': best_option.get('tool', 'respond'),
                    'confidence': learned_pattern['confidence'] * best_score,
                    'reasoning': f'Learned decision (similarity: {best_score:.2f})',
                    'learning_metadata': learned_pattern
                }
        
        return None
    
    async def _weighted_decision(self, options: List[Dict], weights: Dict[str, float]) -> Dict:
        """Make decision using current adaptive weights."""
        best_option = None
        best_score = 0.0
        
        for option in options:
            # Calculate weighted score
            score = 0.0
            score += option.get('confidence', 0.5) * weights.get('confidence', 0.4)
            score += (1.0 - option.get('execution_time', 100) / 1000) * weights.get('speed', 0.3)
            score += option.get('quality_estimate', 0.7) * weights.get('quality', 0.3)
            
            if score > best_score:
                best_option = option
                best_score = score
        
        return {
            'selected_option': best_option.get('tool', 'respond') if best_option else 'respond',
            'confidence': best_score,
            'reasoning': f'Adaptive weighted decision (score: {best_score:.2f})',
            'weights_used': weights.copy()
        }
    
    async def _update_decision_learning(self, decision_record: Dict):
        """Update learning based on decision outcome."""
        outcome_score = decision_record.get('outcome_score', 0.5)
        
        # Update success patterns
        context_key = decision_record['context_key']
        if context_key not in self.context_decision_mapping:
            self.context_decision_mapping[context_key] = {
                'best_option': decision_record['decision']['selected_option'],
                'confidence': outcome_score,
                'sample_count': 1,
                'avg_outcome': outcome_score
            }
        else:
            mapping = self.context_decision_mapping[context_key]
            mapping['sample_count'] += 1
            mapping['avg_outcome'] = (
                (mapping['avg_outcome'] * (mapping['sample_count'] - 1) + outcome_score) / 
                mapping['sample_count']
            )
            
            # Update confidence based on outcomes
            mapping['confidence'] = min(0.95, mapping['avg_outcome'] + 
                                      (mapping['sample_count'] / 100))
        
        # Adapt criteria weights based on what worked
        feedback = decision_record.get('feedback', {})
        if feedback:
            await self._adapt_criteria_weights(feedback, outcome_score)
        
        self.adaptation_count += 1
    
    async def _adapt_criteria_weights(self, feedback: Dict, outcome_score: float):
        """Adapt decision criteria weights based on feedback."""
        # Simple adaptation: increase weight of criteria that led to good outcomes
        adaptation_rate = self.learning_rate * outcome_score
        
        if 'speed_important' in feedback and feedback['speed_important']:
            self.criteria_weights['speed'] += adaptation_rate
        
        if 'quality_important' in feedback and feedback['quality_important']:
            self.criteria_weights['quality'] += adaptation_rate
        
        if 'confidence_important' in feedback and feedback['confidence_important']:
            self.criteria_weights['confidence'] += adaptation_rate
        
        # Normalize weights
        total_weight = sum(self.criteria_weights.values())
        if total_weight > 0:
            for key in self.criteria_weights:
                self.criteria_weights[key] /= total_weight
    
    def _extract_decision_context(self, features: Dict, context: Dict) -> Dict:
        """Extract context for decision learning."""
        return {
            'feature_confidence': features.get('confidence', 0.5),
            'context_size': len(context),
            'time_of_day': time.localtime().tm_hour,
            'has_memory_results': 'memory_results' in context,
            'has_patterns': 'patterns' in features
        }
    
    def _hash_decision_context(self, context: Dict) -> str:
        """Create hash for decision context."""
        return f"conf_{int(context['feature_confidence']*10)}_ctx_{context['context_size']//5}_mem_{context['has_memory_results']}"
    
    def _calculate_option_similarity(self, option1: Dict, option2: str) -> float:
        """Calculate similarity between decision options."""
        if option1.get('tool') == option2:
            return 1.0
        
        # Simple similarity based on tool type
        tool1 = option1.get('tool', '')
        tool2 = option2
        
        if tool1 == tool2:
            return 1.0
        elif (tool1 in ['search', 'find'] and tool2 in ['search', 'find']) or \
             (tool1 in ['analyze', 'explain'] and tool2 in ['analyze', 'explain']):
            return 0.8
        else:
            return 0.3
    
    def _record_decision(self, context: Dict, decision: Dict, options: List[Dict], execution_time: float):
        """Record decision for learning."""
        decision_id = f"decision_{int(time.time() * 1000)}"
        
        self.decision_history.append({
            'decision_id': decision_id,
            'timestamp': time.time(),
            'context': context,
            'context_key': self._hash_decision_context(context),
            'decision': decision,
            'options': options,
            'execution_time': execution_time
        })
        
        # Keep only recent history
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-800:]
    
    async def get_learning_summary(self) -> Dict:
        """Get summary of decision learning."""
        return {
            'total_decisions': len(self.decision_history),
            'adaptations': self.adaptation_count,
            'current_criteria_weights': self.criteria_weights.copy(),
            'learned_contexts': len(self.context_decision_mapping),
            'recent_decisions': [
                {
                    'selected_option': d['decision']['selected_option'],
                    'confidence': d['decision']['confidence'],
                    'outcome_score': d.get('outcome_score', 'pending')
                }
                for d in self.decision_history[-5:]
            ]
        }


async def demo_adaptive_learning():
    """Demonstrate adaptive learning protocols in action."""
    
    print("🧠 ADAPTIVE LEARNING PROTOCOLS DEMO")
    print("=" * 60)
    print("Features:")
    print("• Learning Pattern Selection - Evolves based on success")
    print("• Adaptive Decision Engine - Improves through feedback") 
    print("• Success Pattern Mining - Discovers what works")
    print("• Performance Analytics - Tracks protocol effectiveness")
    print()
    
    # Create adaptive protocols
    pattern_learner = LearningPatternSelection(learning_rate=0.2)
    decision_learner = AdaptiveDecisionEngine(learning_rate=0.15)
    
    # Simulate learning over time
    test_scenarios = [
        {
            'query': 'analyze bee colony behavior patterns',
            'context': {'domain': 'beekeeping', 'user_expertise': 'intermediate'},
            'complexity': ComplexityLevel.FULL,
            'expected_success': 0.8
        },
        {
            'query': 'search for hive management tips',
            'context': {'domain': 'beekeeping', 'user_expertise': 'beginner'},
            'complexity': ComplexityLevel.FAST,
            'expected_success': 0.9
        },
        {
            'query': 'create innovative pollination strategies',
            'context': {'domain': 'research', 'user_expertise': 'expert'},
            'complexity': ComplexityLevel.RESEARCH,
            'expected_success': 0.7
        },
        {
            'query': 'explain simple bee biology',
            'context': {'domain': 'education', 'user_expertise': 'beginner'},
            'complexity': ComplexityLevel.LITE,
            'expected_success': 0.95
        }
    ]
    
    print("🎓 LEARNING SIMULATION (Training Phase)")
    print("-" * 40)
    
    # Phase 1: Initial learning
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}/3:")
        
        for i, scenario in enumerate(test_scenarios):
            # Pattern selection learning
            pattern_result = await pattern_learner.select_pattern(
                scenario['query'],
                scenario['context'], 
                scenario['complexity']
            )
            
            # Decision engine learning
            decision_options = [
                {'tool': 'analyze', 'confidence': 0.8, 'execution_time': 200},
                {'tool': 'search', 'confidence': 0.9, 'execution_time': 50},
                {'tool': 'create', 'confidence': 0.6, 'execution_time': 400}
            ]
            
            decision_result = await decision_learner.make_decision(
                {'confidence': pattern_result['confidence']},
                scenario['context'],
                decision_options
            )
            
            # Simulate success feedback with some noise
            base_success = scenario['expected_success']
            noise = (hash(scenario['query'] + str(epoch)) % 21 - 10) / 100  # ±0.1 noise
            actual_success = max(0.0, min(1.0, base_success + noise))
            
            # Record learning
            query_features = pattern_learner._extract_query_features(
                scenario['query'], scenario['context']
            )
            
            await pattern_learner.record_pattern_success(
                pattern_result['selected_pattern'],
                query_features,
                actual_success
            )
            
            print(f"  Scenario {i+1}: Pattern={pattern_result['selected_pattern'][:15]}..., "
                  f"Decision={decision_result['selected_option']}, Success={actual_success:.2f}")
    
    print(f"\n🎯 LEARNING RESULTS")
    print("-" * 40)
    
    # Show learning summaries
    pattern_summary = await pattern_learner.get_learning_summary()
    decision_summary = await decision_learner.get_learning_summary()
    
    print(f"\n📊 Pattern Selection Learning:")
    print(f"  Patterns Learned: {pattern_summary['total_patterns_learned']}")
    print(f"  Total Adaptations: {pattern_summary['total_adaptations']}")
    print(f"  Avg Execution Time: {pattern_summary['performance_metrics']['avg_execution_time']:.1f}ms")
    
    if pattern_summary['emergent_patterns']:
        print(f"  Emergent Patterns:")
        for name, data in list(pattern_summary['emergent_patterns'].items())[:3]:
            print(f"    • {name}: confidence={data['confidence']:.2f}, samples={data['sample_count']}")
    
    print(f"\n🎯 Decision Engine Learning:")
    print(f"  Total Decisions: {decision_summary['total_decisions']}")
    print(f"  Adaptations: {decision_summary['adaptations']}")
    print(f"  Learned Contexts: {decision_summary['learned_contexts']}")
    print(f"  Current Weights: {decision_summary['current_criteria_weights']}")
    
    print(f"\n🧪 ADAPTIVE TESTING (Using Learned Knowledge)")
    print("-" * 40)
    
    # Phase 2: Test learned knowledge
    for i, scenario in enumerate(test_scenarios[:2]):  # Test first 2 scenarios
        print(f"\nTest {i+1}: {scenario['query'][:50]}...")
        
        # Test learned pattern selection
        pattern_result = await pattern_learner.select_pattern(
            scenario['query'],
            scenario['context'],
            scenario['complexity']
        )
        
        # Test learned decision making
        decision_result = await decision_learner.make_decision(
            {'confidence': pattern_result['confidence']},
            scenario['context'],
            decision_options
        )
        
        print(f"  Pattern: {pattern_result['selected_pattern']} "
              f"(source: {pattern_result.get('source', 'unknown')}, "
              f"confidence: {pattern_result['confidence']:.2f})")
        
        print(f"  Decision: {decision_result['selected_option']} "
              f"(source: {decision_result.get('source', 'unknown')}, "
              f"confidence: {decision_result['confidence']:.2f})")
        
        if 'learning_metadata' in pattern_result:
            meta = pattern_result['learning_metadata']
            print(f"  Learning Info: success_rate={meta.get('success_rate', 0):.2f}, "
                  f"samples={meta.get('sample_count', 0)}")
    
    print(f"\n✨ LEARNING BENEFITS:")
    print(f"• Protocols adapt to user preferences and success patterns")
    print(f"• Performance improves over time through continuous learning")
    print(f"• Emergent patterns discovered automatically")
    print(f"• Context-aware decision making based on historical success")
    print(f"• Multi-criteria optimization that evolves")


if __name__ == "__main__":
    asyncio.run(demo_adaptive_learning())