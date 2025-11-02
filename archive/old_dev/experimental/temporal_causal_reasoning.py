#!/usr/bin/env python3
"""
Temporal Causal Reasoning for mythRL
====================================
Advanced time-aware cause-effect analysis with temporal reasoning.

Core Features:
1. Temporal Event Chains - Track cause-effect sequences across time
2. Causal Graph Construction - Build dynamic causal networks
3. Counterfactual Analysis - "What if" scenarios with temporal implications
4. Temporal Pattern Recognition - Identify recurring causal patterns
5. Future Impact Prediction - Predict effects of current causes
6. Causal Inference Engine - Deduce hidden causal relationships
7. Temporal Consistency Checking - Ensure logical temporal ordering

Philosophy: Understanding not just what happens, but why it happens when it does,
and what effects will ripple through time. Causal reasoning that respects temporal
constraints and discovers hidden temporal dependencies.
"""

import asyncio
import time
import math
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import statistics

# Import quantum synthesis base
try:
    from dev.protocol_modules_mythrl import ComplexityLevel
except ImportError:
    from enum import Enum
    class ComplexityLevel(Enum):
        LITE = "lite"
        FAST = "fast"
        FULL = "full"
        RESEARCH = "research"


class CausalType(Enum):
    """Types of causal relationships."""
    DIRECT = "direct"           # A directly causes B
    INDIRECT = "indirect"       # A causes B through intermediates
    NECESSARY = "necessary"     # A is necessary for B
    SUFFICIENT = "sufficient"   # A is sufficient for B
    CONTRIBUTING = "contributing"  # A contributes to B (partial cause)
    PREVENTING = "preventing"   # A prevents B
    ENABLING = "enabling"       # A enables B but doesn't cause it


class TemporalRelation(Enum):
    """Temporal relationships between events."""
    BEFORE = "before"           # A happens before B
    AFTER = "after"             # A happens after B
    DURING = "during"           # A happens during B
    OVERLAPS = "overlaps"       # A and B overlap in time
    SIMULTANEOUS = "simultaneous"  # A and B happen at same time
    PRECEDES = "precedes"       # A immediately precedes B
    FOLLOWS = "follows"         # A immediately follows B


@dataclass
class TemporalEvent:
    """An event with temporal and causal information."""
    event_id: str
    description: str
    timestamp: float
    duration: float = 0.0
    confidence: float = 1.0
    event_type: str = "general"
    metadata: Dict = field(default_factory=dict)
    
    @property
    def end_time(self) -> float:
        """End time of the event."""
        return self.timestamp + self.duration
    
    def overlaps_with(self, other: 'TemporalEvent') -> bool:
        """Check if this event overlaps with another."""
        return (self.timestamp < other.end_time and 
                self.end_time > other.timestamp)


@dataclass
class CausalLink:
    """A causal relationship between events."""
    cause_id: str
    effect_id: str
    causal_type: CausalType
    temporal_relation: TemporalRelation
    strength: float  # 0.0-1.0
    delay: float = 0.0  # Time delay between cause and effect
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    discovered_at: float = field(default_factory=time.time)
    
    def is_valid_temporal_order(self, cause_event: TemporalEvent, 
                              effect_event: TemporalEvent) -> bool:
        """Check if causal link respects temporal constraints."""
        if self.temporal_relation == TemporalRelation.BEFORE:
            return cause_event.end_time <= effect_event.timestamp
        elif self.temporal_relation == TemporalRelation.PRECEDES:
            return abs(cause_event.end_time - effect_event.timestamp) < 0.1
        elif self.temporal_relation == TemporalRelation.SIMULTANEOUS:
            return cause_event.overlaps_with(effect_event)
        return True


@dataclass
class CausalChain:
    """A sequence of causally connected events."""
    chain_id: str
    events: List[TemporalEvent]
    causal_links: List[CausalLink]
    chain_strength: float = 1.0
    temporal_span: float = 0.0
    pattern_type: str = "linear"
    
    def calculate_chain_strength(self) -> float:
        """Calculate overall strength of causal chain."""
        if not self.causal_links:
            return 0.0
        return statistics.mean(link.strength for link in self.causal_links)
    
    def get_temporal_span(self) -> float:
        """Get total time span of the causal chain."""
        if not self.events:
            return 0.0
        start_time = min(event.timestamp for event in self.events)
        end_time = max(event.end_time for event in self.events)
        return end_time - start_time


class TemporalCausalReasoner:
    """
    Advanced Temporal Causal Reasoning Engine
    
    Features:
    - Constructs temporal causal graphs from events
    - Identifies causal patterns across time
    - Performs counterfactual analysis
    - Predicts future effects of current causes
    - Validates temporal consistency of causal claims
    """
    
    def __init__(self, temporal_window: float = 3600.0):  # 1 hour default window
        self.temporal_window = temporal_window
        self.events = {}  # event_id -> TemporalEvent
        self.causal_links = []  # List of CausalLink
        self.causal_chains = {}  # chain_id -> CausalChain
        self.temporal_patterns = defaultdict(list)  # pattern_type -> chains
        self.reasoning_history = []
        self.counterfactual_cache = {}
        
        # Reasoning parameters by complexity
        self.reasoning_config = {
            ComplexityLevel.LITE: {
                'max_chain_length': 3,
                'temporal_precision': 1.0,
                'causal_threshold': 0.6
            },
            ComplexityLevel.FAST: {
                'max_chain_length': 5,
                'temporal_precision': 0.5,
                'causal_threshold': 0.5
            },
            ComplexityLevel.FULL: {
                'max_chain_length': 8,
                'temporal_precision': 0.1,
                'causal_threshold': 0.4
            },
            ComplexityLevel.RESEARCH: {
                'max_chain_length': 15,
                'temporal_precision': 0.01,
                'causal_threshold': 0.3
            }
        }
    
    async def add_event(self, event: TemporalEvent) -> str:
        """Add a temporal event to the reasoning system."""
        self.events[event.event_id] = event
        
        # Automatically discover causal relationships
        await self._discover_causal_relationships(event)
        
        # Update causal chains
        await self._update_causal_chains(event)
        
        return event.event_id
    
    async def analyze_causal_chain(self, start_event_id: str, 
                                 complexity: ComplexityLevel) -> Dict:
        """Analyze causal chains starting from a given event."""
        
        if start_event_id not in self.events:
            return {'error': 'Event not found'}
        
        config = self.reasoning_config[complexity]
        start_time = time.perf_counter()
        
        print(f"🔗 Analyzing causal chain from: {start_event_id}")
        print(f"   Complexity: {complexity.name}")
        print(f"   Max chain length: {config['max_chain_length']}")
        
        # Find all causal chains starting from this event
        chains = await self._find_causal_chains(
            start_event_id, 
            config['max_chain_length'],
            config['causal_threshold']
        )
        
        # Analyze temporal patterns
        patterns = await self._analyze_temporal_patterns(chains)
        
        # Predict future effects
        future_predictions = await self.predict_future_effects(
            [start_event_id], 3600, complexity  # 1 hour prediction horizon
        )
        
        # Calculate reasoning confidence
        confidence = await self._calculate_reasoning_confidence(chains)
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        result = {
            'start_event': self.events[start_event_id].description,
            'causal_chains_found': len(chains),
            'strongest_chain': self._get_strongest_chain(chains),
            'temporal_patterns': patterns,
            'future_predictions': future_predictions,
            'reasoning_confidence': confidence,
            'analysis_metadata': {
                'complexity_level': complexity.name,
                'execution_time_ms': execution_time,
                'events_analyzed': len(self.events),
                'causal_links_evaluated': len(self.causal_links)
            }
        }
        
        print(f"   ✨ Analysis complete in {execution_time:.1f}ms")
        print(f"   📊 Found {len(chains)} causal chains")
        
        return result
    
    async def counterfactual_analysis(self, event_id: str, 
                                    counterfactual_scenario: Dict,
                                    complexity: ComplexityLevel) -> Dict:
        """Perform 'what if' counterfactual analysis."""
        
        print(f"🔮 Counterfactual analysis: {event_id}")
        print(f"   Scenario: {counterfactual_scenario.get('description', 'Unknown')}")
        
        if event_id not in self.events:
            return {'error': 'Event not found'}
        
        original_event = self.events[event_id]
        config = self.reasoning_config[complexity]
        
        # Create counterfactual scenario
        counterfactual_event = TemporalEvent(
            event_id=f"{event_id}_counterfactual",
            description=counterfactual_scenario.get('description', original_event.description),
            timestamp=counterfactual_scenario.get('timestamp', original_event.timestamp),
            duration=counterfactual_scenario.get('duration', original_event.duration),
            confidence=0.8,  # Counterfactuals are inherently uncertain
            event_type="counterfactual"
        )
        
        # Temporarily add counterfactual event
        self.events[counterfactual_event.event_id] = counterfactual_event
        
        try:
            # Analyze chains from counterfactual
            counterfactual_chains = await self._find_causal_chains(
                counterfactual_event.event_id,
                config['max_chain_length'],
                config['causal_threshold']
            )
            
            # Compare with original chains
            original_chains = await self._find_causal_chains(
                event_id,
                config['max_chain_length'],
                config['causal_threshold']
            )
            
            # Calculate differences
            differences = await self._compare_causal_outcomes(
                original_chains, counterfactual_chains
            )
            
            result = {
                'original_event': original_event.description,
                'counterfactual_scenario': counterfactual_scenario,
                'original_chains': len(original_chains),
                'counterfactual_chains': len(counterfactual_chains),
                'outcome_differences': differences,
                'impact_assessment': await self._assess_counterfactual_impact(differences),
                'confidence': min(0.9, statistics.mean([
                    chain.chain_strength for chain in counterfactual_chains
                ]) if counterfactual_chains else 0.5)
            }
            
        finally:
            # Remove counterfactual event
            del self.events[counterfactual_event.event_id]
        
        print(f"   🎯 Impact: {result['impact_assessment']['overall_impact']}")
        
        return result
    
    async def predict_future_effects(self, current_events: List[str],
                                   prediction_horizon: float,
                                   complexity: ComplexityLevel) -> Dict:
        """Predict future effects based on current events."""
        
        print(f"🔭 Predicting future effects")
        print(f"   Events: {len(current_events)}")
        print(f"   Horizon: {prediction_horizon}s")
        
        config = self.reasoning_config[complexity]
        current_time = time.time()
        
        predictions = []
        confidence_scores = []
        
        for event_id in current_events:
            if event_id not in self.events:
                continue
            
            # Find causal chains from this event
            chains = await self._find_causal_chains(
                event_id, config['max_chain_length'], config['causal_threshold']
            )
            
            # Predict effects within time horizon
            event_predictions = await self._predict_chain_effects(
                chains, current_time, prediction_horizon
            )
            
            predictions.extend(event_predictions)
            
            # Calculate prediction confidence
            chain_confidences = [chain.chain_strength for chain in chains]
            if chain_confidences:
                confidence_scores.append(statistics.mean(chain_confidences))
        
        # Aggregate and deduplicate predictions
        aggregated_predictions = await self._aggregate_predictions(predictions)
        
        overall_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.5
        
        result = {
            'prediction_horizon': prediction_horizon,
            'current_events_analyzed': len(current_events),
            'predicted_effects': aggregated_predictions,
            'prediction_confidence': overall_confidence,
            'temporal_distribution': await self._analyze_prediction_timing(aggregated_predictions),
            'complexity_level': complexity.name
        }
        
        print(f"   📈 Predicted {len(aggregated_predictions)} future effects")
        print(f"   🎯 Confidence: {overall_confidence:.2f}")
        
        return result
    
    async def validate_temporal_consistency(self, proposed_chain: List[Dict]) -> Dict:
        """Validate temporal consistency of a proposed causal chain."""
        
        print(f"⏰ Validating temporal consistency")
        print(f"   Chain length: {len(proposed_chain)}")
        
        validation_results = {
            'is_consistent': True,
            'violations': [],
            'consistency_score': 1.0,
            'temporal_gaps': [],
            'suggestions': []
        }
        
        for i in range(len(proposed_chain) - 1):
            current = proposed_chain[i]
            next_event = proposed_chain[i + 1]
            
            # Check temporal ordering
            if current.get('timestamp', 0) > next_event.get('timestamp', 0):
                validation_results['violations'].append({
                    'type': 'temporal_order',
                    'description': f"Event {i+1} occurs before event {i}",
                    'severity': 'high'
                })
                validation_results['is_consistent'] = False
            
            # Check causal delay consistency
            delay = next_event.get('timestamp', 0) - current.get('timestamp', 0)
            expected_delay = current.get('expected_delay', 0)
            
            if abs(delay - expected_delay) > 1.0:  # 1 second tolerance
                validation_results['temporal_gaps'].append({
                    'between_events': f"{i} -> {i+1}",
                    'actual_delay': delay,
                    'expected_delay': expected_delay,
                    'gap_size': abs(delay - expected_delay)
                })
        
        # Calculate consistency score
        violation_count = len(validation_results['violations'])
        gap_count = len(validation_results['temporal_gaps'])
        
        validation_results['consistency_score'] = max(0.0, 1.0 - 
            (violation_count * 0.2 + gap_count * 0.1))
        
        # Generate suggestions
        if not validation_results['is_consistent']:
            validation_results['suggestions'] = await self._generate_consistency_suggestions(
                validation_results['violations']
            )
        
        print(f"   ✅ Consistency: {validation_results['is_consistent']}")
        print(f"   📊 Score: {validation_results['consistency_score']:.2f}")
        
        return validation_results
    
    async def _discover_causal_relationships(self, new_event: TemporalEvent):
        """Discover causal relationships involving the new event."""
        
        for existing_id, existing_event in self.events.items():
            if existing_id == new_event.event_id:
                continue
            
            # Check for potential causal relationships
            causal_link = await self._analyze_potential_causality(
                existing_event, new_event
            )
            
            if causal_link and causal_link.strength > 0.3:
                self.causal_links.append(causal_link)
    
    async def _analyze_potential_causality(self, event1: TemporalEvent, 
                                         event2: TemporalEvent) -> Optional[CausalLink]:
        """Analyze potential causal relationship between two events."""
        
        # Determine temporal relationship
        temporal_rel = self._determine_temporal_relationship(event1, event2)
        
        # Calculate causal strength based on various factors
        strength = 0.0
        
        # Temporal proximity increases causal likelihood
        time_diff = abs(event2.timestamp - event1.end_time)
        if time_diff < 3600:  # Within 1 hour
            strength += 0.3 * (1.0 - time_diff / 3600)
        
        # Event type similarity
        if event1.event_type == event2.event_type:
            strength += 0.2
        
        # Content similarity (simple keyword matching)
        content_similarity = self._calculate_content_similarity(
            event1.description, event2.description
        )
        strength += content_similarity * 0.3
        
        # Confidence boost
        strength *= (event1.confidence + event2.confidence) / 2
        
        if strength > 0.3:
            # Determine causal type based on strength and temporal relation
            if strength > 0.7:
                causal_type = CausalType.DIRECT
            elif strength > 0.5:
                causal_type = CausalType.CONTRIBUTING
            else:
                causal_type = CausalType.INDIRECT
            
            return CausalLink(
                cause_id=event1.event_id if temporal_rel == TemporalRelation.BEFORE else event2.event_id,
                effect_id=event2.event_id if temporal_rel == TemporalRelation.BEFORE else event1.event_id,
                causal_type=causal_type,
                temporal_relation=temporal_rel,
                strength=strength,
                delay=abs(event2.timestamp - event1.timestamp),
                confidence=min(event1.confidence, event2.confidence)
            )
        
        return None
    
    def _determine_temporal_relationship(self, event1: TemporalEvent, 
                                       event2: TemporalEvent) -> TemporalRelation:
        """Determine temporal relationship between events."""
        
        if abs(event1.timestamp - event2.timestamp) < 0.1:
            return TemporalRelation.SIMULTANEOUS
        elif event1.end_time <= event2.timestamp:
            return TemporalRelation.BEFORE
        elif event2.end_time <= event1.timestamp:
            return TemporalRelation.AFTER
        elif event1.overlaps_with(event2):
            return TemporalRelation.OVERLAPS
        else:
            return TemporalRelation.BEFORE if event1.timestamp < event2.timestamp else TemporalRelation.AFTER
    
    def _calculate_content_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple content similarity between event descriptions."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def _find_causal_chains(self, start_event_id: str, max_length: int, 
                                threshold: float) -> List[CausalChain]:
        """Find causal chains starting from an event."""
        
        chains = []
        
        # Use breadth-first search to find causal chains
        queue = deque([(start_event_id, [start_event_id], [])])
        
        while queue:
            current_id, path, links = queue.popleft()
            
            if len(path) >= max_length:
                continue
            
            # Find effects of current event
            for link in self.causal_links:
                if (link.cause_id == current_id and 
                    link.effect_id not in path and 
                    link.strength >= threshold):
                    
                    new_path = path + [link.effect_id]
                    new_links = links + [link]
                    
                    # Create chain if we have at least 2 events
                    if len(new_path) >= 2:
                        events = [self.events[event_id] for event_id in new_path]
                        chain = CausalChain(
                            chain_id=f"chain_{len(chains)}",
                            events=events,
                            causal_links=new_links
                        )
                        chain.chain_strength = chain.calculate_chain_strength()
                        chain.temporal_span = chain.get_temporal_span()
                        chains.append(chain)
                    
                    # Continue search
                    queue.append((link.effect_id, new_path, new_links))
        
        return chains
    
    async def _predict_chain_effects(self, chains: List[CausalChain], 
                                   current_time: float, horizon: float) -> List[Dict]:
        """Predict future effects from causal chains."""
        
        predictions = []
        
        for chain in chains:
            # Look at the last event in the chain
            last_event = chain.events[-1]
            
            # If the chain is recent, predict continuation
            if last_event.timestamp > current_time - 3600:  # Last hour
                # Predict next effect based on chain pattern
                predicted_time = current_time + statistics.mean([
                    link.delay for link in chain.causal_links
                ])
                
                if predicted_time <= current_time + horizon:
                    predictions.append({
                        'predicted_event': f"Continuation of {last_event.description}",
                        'predicted_time': predicted_time,
                        'confidence': chain.chain_strength * 0.8,  # Reduced for prediction
                        'based_on_chain': chain.chain_id,
                        'reasoning': f"Pattern from {len(chain.events)}-event chain"
                    })
        
        return predictions
    
    def _get_strongest_chain(self, chains: List[CausalChain]) -> Optional[Dict]:
        """Get the strongest causal chain."""
        if not chains:
            return None
        
        strongest = max(chains, key=lambda c: c.chain_strength)
        
        return {
            'chain_id': strongest.chain_id,
            'strength': strongest.chain_strength,
            'event_count': len(strongest.events),
            'temporal_span': strongest.temporal_span,
            'events': [event.description for event in strongest.events],
            'causal_types': [link.causal_type.value for link in strongest.causal_links]
        }
    
    async def _analyze_temporal_patterns(self, chains: List[CausalChain]) -> Dict:
        """Analyze temporal patterns in causal chains."""
        
        patterns = {
            'average_chain_length': statistics.mean([len(chain.events) for chain in chains]) if chains else 0,
            'average_temporal_span': statistics.mean([chain.temporal_span for chain in chains]) if chains else 0,
            'causal_type_distribution': defaultdict(int),
            'temporal_clustering': []
        }
        
        # Analyze causal type distribution
        for chain in chains:
            for link in chain.causal_links:
                patterns['causal_type_distribution'][link.causal_type.value] += 1
        
        return patterns
    
    async def _calculate_reasoning_confidence(self, chains: List[CausalChain]) -> float:
        """Calculate overall confidence in causal reasoning."""
        if not chains:
            return 0.0
        
        # Factor in chain strengths and evidence quality
        chain_strengths = [chain.chain_strength for chain in chains]
        avg_strength = statistics.mean(chain_strengths)
        
        # Factor in number of chains (more chains = more evidence)
        chain_factor = min(1.0, len(chains) / 5.0)  # Normalize to max 5 chains
        
        return avg_strength * chain_factor
    
    async def _update_causal_chains(self, new_event: TemporalEvent):
        """Update existing causal chains with new event."""
        # This is a simplified implementation
        # In a full system, this would update existing chains
        pass
    
    async def _compare_causal_outcomes(self, original_chains: List[CausalChain],
                                     counterfactual_chains: List[CausalChain]) -> Dict:
        """Compare outcomes between original and counterfactual scenarios."""
        
        return {
            'chain_count_difference': len(counterfactual_chains) - len(original_chains),
            'strength_difference': (
                statistics.mean([c.chain_strength for c in counterfactual_chains]) -
                statistics.mean([c.chain_strength for c in original_chains])
            ) if original_chains and counterfactual_chains else 0.0,
            'new_effects': len(counterfactual_chains) - len(original_chains),
            'changed_outcomes': abs(len(counterfactual_chains) - len(original_chains))
        }
    
    async def _assess_counterfactual_impact(self, differences: Dict) -> Dict:
        """Assess the impact of counterfactual changes."""
        
        impact_score = abs(differences.get('strength_difference', 0)) + \
                      abs(differences.get('new_effects', 0)) * 0.3
        
        if impact_score > 0.5:
            impact_level = "high"
        elif impact_score > 0.2:
            impact_level = "medium"
        else:
            impact_level = "low"
        
        return {
            'overall_impact': impact_level,
            'impact_score': impact_score,
            'primary_changes': [
                f"{differences.get('new_effects', 0)} new causal effects",
                f"Strength change: {differences.get('strength_difference', 0):.2f}"
            ]
        }
    
    async def _aggregate_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """Aggregate and deduplicate predictions."""
        # Simple aggregation - in full system would be more sophisticated
        return predictions[:10]  # Limit to top 10 predictions
    
    async def _analyze_prediction_timing(self, predictions: List[Dict]) -> Dict:
        """Analyze temporal distribution of predictions."""
        
        if not predictions:
            return {'no_predictions': True}
        
        times = [p.get('predicted_time', 0) for p in predictions]
        
        return {
            'earliest_prediction': min(times) if times else 0,
            'latest_prediction': max(times) if times else 0,
            'prediction_span': max(times) - min(times) if times else 0,
            'prediction_count': len(predictions)
        }
    
    async def _generate_consistency_suggestions(self, violations: List[Dict]) -> List[str]:
        """Generate suggestions to fix temporal consistency violations."""
        
        suggestions = []
        
        for violation in violations:
            if violation['type'] == 'temporal_order':
                suggestions.append("Reorder events to respect causality")
            elif violation['type'] == 'delay_mismatch':
                suggestions.append("Adjust event timing to match causal delays")
        
        return suggestions
    
    async def get_reasoning_summary(self) -> Dict:
        """Get summary of temporal causal reasoning state."""
        
        return {
            'total_events': len(self.events),
            'total_causal_links': len(self.causal_links),
            'total_chains': len(self.causal_chains),
            'reasoning_operations': len(self.reasoning_history),
            'temporal_window': self.temporal_window,
            'event_types': list(set(event.event_type for event in self.events.values())),
            'causal_type_distribution': {
                causal_type.value: sum(1 for link in self.causal_links 
                                     if link.causal_type == causal_type)
                for causal_type in CausalType
            }
        }


async def demo_temporal_causal_reasoning():
    """Demonstrate temporal causal reasoning capabilities."""
    
    print("⏰ TEMPORAL CAUSAL REASONING DEMO")
    print("=" * 60)
    print("Features:")
    print("• Temporal Event Chains - Track cause-effect sequences")
    print("• Causal Graph Construction - Build dynamic causal networks")
    print("• Counterfactual Analysis - 'What if' scenarios")
    print("• Future Impact Prediction - Predict temporal effects")
    print("• Temporal Consistency Checking - Validate causal logic")
    print()
    
    # Create temporal causal reasoner
    reasoner = TemporalCausalReasoner(temporal_window=7200)  # 2 hour window
    
    # Create a sequence of bee-related events
    current_time = time.time()
    
    events = [
        TemporalEvent(
            event_id="bee_1",
            description="Queen bee starts increased egg laying",
            timestamp=current_time - 3600,  # 1 hour ago
            duration=300,  # 5 minutes
            confidence=0.9,
            event_type="biological"
        ),
        TemporalEvent(
            event_id="bee_2", 
            description="Worker bees increase foraging activity",
            timestamp=current_time - 3300,  # 55 minutes ago
            duration=1800,  # 30 minutes
            confidence=0.8,
            event_type="biological"
        ),
        TemporalEvent(
            event_id="bee_3",
            description="Hive temperature rises due to increased activity",
            timestamp=current_time - 3000,  # 50 minutes ago
            duration=600,  # 10 minutes
            confidence=0.85,
            event_type="environmental"
        ),
        TemporalEvent(
            event_id="bee_4",
            description="Nectar stores in hive increase",
            timestamp=current_time - 2400,  # 40 minutes ago
            duration=900,  # 15 minutes
            confidence=0.9,
            event_type="resource"
        ),
        TemporalEvent(
            event_id="bee_5",
            description="Scout bees discover new flower patch",
            timestamp=current_time - 1800,  # 30 minutes ago
            duration=120,  # 2 minutes
            confidence=0.7,
            event_type="discovery"
        ),
        TemporalEvent(
            event_id="bee_6",
            description="Waggle dance communication increases",
            timestamp=current_time - 1500,  # 25 minutes ago
            duration=300,  # 5 minutes
            confidence=0.8,
            event_type="communication"
        )
    ]
    
    print("🐝 ADDING TEMPORAL EVENTS")
    print("-" * 40)
    
    # Add events to reasoner
    for event in events:
        await reasoner.add_event(event)
        print(f"   Added: {event.description[:50]}... at {event.timestamp:.0f}")
    
    print(f"\n📊 System state: {len(events)} events, {len(reasoner.causal_links)} causal links discovered")
    
    print(f"\n🔗 CAUSAL CHAIN ANALYSIS")
    print("-" * 40)
    
    # Analyze causal chains for different complexity levels
    for complexity in [ComplexityLevel.FAST, ComplexityLevel.FULL, ComplexityLevel.RESEARCH]:
        print(f"\n📋 {complexity.name} Complexity Analysis:")
        
        analysis = await reasoner.analyze_causal_chain("bee_1", complexity)
        
        print(f"   Causal chains found: {analysis['causal_chains_found']}")
        print(f"   Reasoning confidence: {analysis['reasoning_confidence']:.2f}")
        print(f"   Analysis time: {analysis['analysis_metadata']['execution_time_ms']:.1f}ms")
        
        if analysis.get('strongest_chain'):
            chain = analysis['strongest_chain']
            print(f"   Strongest chain: {chain['event_count']} events, strength {chain['strength']:.2f}")
            print(f"   Chain events: {' → '.join(chain['events'][:3])}...")
    
    print(f"\n🔮 COUNTERFACTUAL ANALYSIS")
    print("-" * 40)
    
    # Perform counterfactual analysis
    counterfactual_scenario = {
        'description': "Queen bee reduces egg laying due to stress",
        'timestamp': current_time - 3600,
        'duration': 300
    }
    
    counterfactual = await reasoner.counterfactual_analysis(
        "bee_1", counterfactual_scenario, ComplexityLevel.FULL
    )
    
    print(f"   Original scenario: {counterfactual['original_event']}")
    print(f"   Counterfactual: {counterfactual_scenario['description']}")
    print(f"   Impact assessment: {counterfactual['impact_assessment']['overall_impact']}")
    print(f"   Chain difference: {counterfactual['outcome_differences']['chain_count_difference']}")
    print(f"   Confidence: {counterfactual['confidence']:.2f}")
    
    print(f"\n🔭 FUTURE PREDICTION")
    print("-" * 40)
    
    # Predict future effects
    prediction = await reasoner.predict_future_effects(
        ["bee_5", "bee_6"],  # Recent discovery and communication events
        3600,  # 1 hour into the future
        ComplexityLevel.FULL
    )
    
    print(f"   Prediction horizon: {prediction['prediction_horizon']/60:.0f} minutes")
    print(f"   Events analyzed: {prediction['current_events_analyzed']}")
    print(f"   Predicted effects: {len(prediction['predicted_effects'])}")
    print(f"   Prediction confidence: {prediction['prediction_confidence']:.2f}")
    
    # Show sample predictions
    for i, pred in enumerate(prediction['predicted_effects'][:3]):
        print(f"   Prediction {i+1}: {pred.get('predicted_event', 'Unknown')[:50]}...")
        print(f"                   Time: +{(pred.get('predicted_time', 0) - current_time)/60:.0f}min, "
              f"Confidence: {pred.get('confidence', 0):.2f}")
    
    print(f"\n⏰ TEMPORAL CONSISTENCY VALIDATION")
    print("-" * 40)
    
    # Test temporal consistency validation
    proposed_chain = [
        {'timestamp': current_time - 3600, 'description': 'Event A'},
        {'timestamp': current_time - 3000, 'description': 'Event B'},
        {'timestamp': current_time - 3300, 'description': 'Event C'},  # Out of order!
    ]
    
    validation = await reasoner.validate_temporal_consistency(proposed_chain)
    
    print(f"   Chain is consistent: {validation['is_consistent']}")
    print(f"   Consistency score: {validation['consistency_score']:.2f}")
    print(f"   Violations found: {len(validation['violations'])}")
    
    if validation['violations']:
        for violation in validation['violations']:
            print(f"      {violation['type']}: {violation['description']}")
    
    if validation['suggestions']:
        print(f"   Suggestions: {', '.join(validation['suggestions'])}")
    
    print(f"\n📈 TEMPORAL PATTERNS ANALYSIS")
    print("-" * 40)
    
    summary = await reasoner.get_reasoning_summary()
    print(f"   Total events tracked: {summary['total_events']}")
    print(f"   Causal links discovered: {summary['total_causal_links']}")
    print(f"   Event types: {', '.join(summary['event_types'])}")
    print(f"   Causal type distribution:")
    
    for causal_type, count in summary['causal_type_distribution'].items():
        if count > 0:
            print(f"      {causal_type}: {count}")
    
    print(f"\n⚡ REAL-TIME CAUSAL DISCOVERY")
    print("-" * 40)
    
    # Add a new event and show real-time discovery
    new_event = TemporalEvent(
        event_id="bee_7",
        description="Hive swarms as population reaches capacity",
        timestamp=current_time - 600,  # 10 minutes ago
        duration=1800,  # 30 minutes
        confidence=0.95,
        event_type="biological"
    )
    
    print(f"   Adding new event: {new_event.description}")
    await reasoner.add_event(new_event)
    
    # Analyze immediate causal impact
    new_analysis = await reasoner.analyze_causal_chain("bee_7", ComplexityLevel.FAST)
    
    print(f"   New causal chains: {new_analysis['causal_chains_found']}")
    print(f"   Updated total links: {len(reasoner.causal_links)}")
    print(f"   Real-time discovery: {len(reasoner.causal_links) - summary['total_causal_links']} new links")
    
    print(f"\n✨ TEMPORAL CAUSAL REASONING BENEFITS:")
    print(f"• Discovers hidden cause-effect relationships across time")
    print(f"• Validates temporal consistency of causal claims")
    print(f"• Enables 'what if' counterfactual analysis")
    print(f"• Predicts future effects based on causal patterns")
    print(f"• Builds dynamic causal networks from temporal events")
    print(f"• Adapts reasoning complexity based on analysis needs")


if __name__ == "__main__":
    asyncio.run(demo_temporal_causal_reasoning())