#!/usr/bin/env python3
"""
Protocol + Modules = mythRL Architecture
========================================
Complete architectural specification for the new mythRL system.

Core Philosophy:
- Shuttle = Creative Orchestrator (Internal Intelligence)
- Protocols = Interface Contracts (Swappable Implementations)
- Modules = Domain-Specific Implementations
- 3-5-7-9 = Progressive Complexity Activation

This architecture enables:
1. Clean separation of concerns
2. Swappable implementations
3. Progressive complexity scaling
4. Full computational provenance
5. Creative combinatorial intelligence

Architecture Diagram:
                    ┌─────────────────────┐
                    │      SHUTTLE        │
                    │ (Creative Orchestr.) │
                    │ ┌─────────────────┐ │
                    │ │ Synthesis Bridge│ │
                    │ │ Temporal Windows│ │
                    │ │ Spacetime Trace │ │
                    │ │ Routing System  │ │
                    │ └─────────────────┘ │
                    └──┬──┬──┬──┬──┬──┬───┘
                       │  │  │  │  │  │
            ┌──────────▼┐ │  │  │  │  │
            │ Pattern   │ │  │  │  │  │
            │ Selection │ │  │  │  │  │
            │ Protocol  │ │  │  │  │  │
            └───────────┘ │  │  │  │  │
               ┌──────────▼┐ │  │  │  │
               │ Decision  │ │  │  │  │
               │ Engine    │ │  │  │  │
               │ Protocol  │ │  │  │  │
               └───────────┘ │  │  │  │
                  ┌──────────▼┐ │  │  │
                  │ Memory    │ │  │  │
                  │ Backend   │ │  │  │
                  │ Protocol  │ │  │  │
                  └───────────┘ │  │  │
                     ┌──────────▼┐ │  │
                     │ Feature   │ │  │
                     │ Extraction│ │  │
                     │ Protocol  │ │  │
                     └───────────┘ │  │
                        ┌──────────▼┐ │
                        │ WarpSpace │ │
                        │ Protocol  │ │
                        └───────────┘ │
                           ┌──────────▼┐
                           │ Tool      │
                           │ Execution │
                           │ Protocol  │
                           └───────────┘
"""

from typing import Protocol, Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import time

# Import mythRL core types and protocols from HoloLoom
from HoloLoom.protocols import (
    ComplexityLevel,
    ProvenceTrace,
    MythRLResult,
    PatternSelectionProtocol,
    DecisionEngineProtocol,
    FeatureExtractionProtocol,
    WarpSpaceProtocol,
    MemoryStore as MemoryBackendProtocol,  # Alias for compatibility
    ToolExecutor as ToolExecutionProtocol,  # Alias for compatibility
)


# Note: Protocol definitions have been moved to HoloLoom.protocols
# This file now focuses on the MythRLShuttle implementation

# === SHUTTLE: CREATIVE ORCHESTRATOR ===

class MythRLShuttle:
    """
    The Shuttle: Creative orchestrator with internal intelligence.
    
    Contains all coordination logic:
    - Synthesis Bridge: Pattern integration across modules
    - Temporal Windows: Time-aware processing coordination  
    - Spacetime Tracing: Full computational provenance
    - Routing System: Intelligent module activation
    
    Philosophy: Protocol + Modules = mythRL
    """
    
    def __init__(self):
        # Protocol implementations (swappable)
        self.pattern_selection: Optional[PatternSelectionProtocol] = None
        self.decision_engine: Optional[DecisionEngineProtocol] = None
        self.memory_backend: Optional[MemoryBackendProtocol] = None
        self.feature_extraction: Optional[FeatureExtractionProtocol] = None
        self.warp_space: Optional[WarpSpaceProtocol] = None
        self.tool_execution: Optional[ToolExecutionProtocol] = None
        
        # Internal shuttle intelligence
        self.routing_patterns = self._initialize_routing_patterns()
        self.complexity_thresholds = self._initialize_complexity_thresholds()
    
    def register_protocol(self, protocol_name: str, implementation: Any):
        """Register a protocol implementation."""
        setattr(self, protocol_name, implementation)
        print(f"✅ Registered {protocol_name} protocol implementation")
    
    async def weave(self, query: str, context: Optional[Dict] = None) -> MythRLResult:
        """
        Main weaving operation - orchestrates entire mythRL pipeline.
        
        This method contains the creative intelligence that combines:
        - Synthesis Bridge logic
        - Temporal Window creation
        - Spacetime Tracing
        - Progressive complexity activation
        """
        operation_id = f"weave_{int(time.time() * 1000)}"
        start_time = time.perf_counter()
        context = context or {}
        
        # Initialize provenance trace
        trace = ProvenceTrace(
            operation_id=operation_id,
            complexity_level=ComplexityLevel.LITE,
            start_time=start_time
        )
        
        trace.add_shuttle_event("weave_start", f"Starting weave operation for query: {query[:50]}...")
        
        # === SHUTTLE'S ROUTING INTELLIGENCE ===
        complexity_level = await self._assess_complexity_needed(query, context, trace)
        trace.complexity_level = complexity_level
        
        print(f"🚀 MythRL Shuttle Activated - Complexity: {complexity_level.name} ({complexity_level.value} steps)")
        
        # === PROGRESSIVE COMPLEXITY EXECUTION ===
        
        # LITE (3 steps): Extract → Route → Execute  
        result_data = await self._execute_lite_path(query, context, trace)
        
        if complexity_level.value >= 5:
            # FAST (5 steps): + Pattern Selection + Temporal Windows
            result_data = await self._enhance_fast_path(result_data, query, context, trace)
        
        if complexity_level.value >= 7:
            # FULL (7 steps): + Decision Engine + Synthesis Bridge
            result_data = await self._enhance_full_path(result_data, query, context, trace)
        
        if complexity_level.value >= 9:
            # RESEARCH (9 steps): + Advanced WarpSpace + Full Tracing
            result_data = await self._enhance_research_path(result_data, query, context, trace)
        
        # === FINALIZE WITH SPACETIME COORDINATES ===
        spacetime_coords = await self._compute_spacetime_coordinates(result_data, trace)
        
        # Create final result
        final_result = MythRLResult(
            query=query,
            output=result_data.get('output', 'No output generated'),
            confidence=result_data.get('confidence', 0.5),
            complexity_level=complexity_level,
            provenance=trace,
            spacetime_coordinates=spacetime_coords
        )
        
        trace.add_shuttle_event("weave_complete", "Weaving operation completed", {
            'final_confidence': final_result.confidence,
            'total_duration_ms': (time.perf_counter() - start_time) * 1000
        })
        
        return final_result
    
    # === INTERNAL SHUTTLE INTELLIGENCE ===
    
    async def _assess_complexity_needed(self, query: str, context: Dict, trace: ProvenceTrace) -> ComplexityLevel:
        """Shuttle's routing intelligence - determines complexity level."""
        query_lower = query.lower()
        
        # Pattern-based assessment with priorities
        complexity_score = 0.0
        
        # Research indicators (highest weight)
        research_patterns = ["research", "innovate", "experiment", "complex", "multi-dimensional", "optimization", "advanced"]
        research_matches = sum(1 for pattern in research_patterns if pattern in query_lower)
        complexity_score += research_matches * 0.4
        
        # Analysis indicators (high weight)
        analysis_patterns = ["analyze", "relationship", "explain", "compare", "design", "between", "factors", "patterns"]
        analysis_matches = sum(1 for pattern in analysis_patterns if pattern in query_lower)
        complexity_score += analysis_matches * 0.3
        
        # Search indicators (medium weight)
        search_patterns = ["search", "find", "show", "list", "lookup", "get"]
        search_matches = sum(1 for pattern in search_patterns if pattern in query_lower)
        complexity_score += search_matches * 0.2
        
        # Length and structure assessment
        word_count = len(query.split())
        complexity_score += min(0.5, word_count / 20)  # Normalize word count
        
        # Map score to complexity level
        if complexity_score >= 1.5:
            level = ComplexityLevel.RESEARCH
        elif complexity_score >= 1.0:
            level = ComplexityLevel.FULL
        elif complexity_score >= 0.5:
            level = ComplexityLevel.FAST
        else:
            level = ComplexityLevel.LITE
        
        trace.add_shuttle_event("complexity_assessment", f"Assessed complexity: {level.name}", {
            'complexity_score': complexity_score,
            'word_count': word_count,
            'research_matches': research_matches,
            'analysis_matches': analysis_matches,
            'search_matches': search_matches
        })
        
        return level
    
    async def _execute_lite_path(self, query: str, context: Dict, trace: ProvenceTrace) -> Dict:
        """
        LITE (3 steps): Essential processing only
        1. Feature Extraction (basic)
        2. Memory Retrieval (simple)  
        3. Tool Execution (direct)
        """
        print("⚡ LITE Path: Essential processing")
        result = {'confidence': 0.0, 'outputs': []}
        
        # Step 1: Basic Feature Extraction
        if self.feature_extraction:
            start_time = time.perf_counter()
            features = await self.feature_extraction.extract_features(query, scales=[96])
            duration = (time.perf_counter() - start_time) * 1000
            trace.add_protocol_call("feature_extraction", "extract_features", duration, f"Extracted 96d features")
            trace.modules_invoked.append("feature_extraction")
            result['features'] = features
            result['confidence'] = max(result['confidence'], features.get('confidence', 0.5))
        
        # Step 2: Intelligent Memory Retrieval with Multipass Crawling
        if self.memory_backend:
            start_time = time.perf_counter()
            memory_results = await self._multipass_memory_crawl(query, trace.complexity_level, trace)
            duration = (time.perf_counter() - start_time) * 1000
            trace.add_protocol_call("memory_backend", "multipass_crawl", duration, 
                                   f"Multipass crawl: {memory_results['total_items']} items across {memory_results['crawl_depth']} depths")
            trace.modules_invoked.append("memory_backend")
            result['memory_results'] = memory_results['results']
            result['crawl_stats'] = memory_results['crawl_stats']
            if memory_results['results']:
                # Higher confidence for multipass results
                result['confidence'] = max(result['confidence'], memory_results['fusion_score'])
        
        # Step 3: Direct Tool Execution (Shuttle's internal routing)
        tool_result = await self._execute_direct_routing(query, result, trace)
        result['tool_result'] = tool_result
        result['output'] = tool_result.get('output', f"Processed: {query}")
        result['confidence'] = max(result['confidence'], tool_result.get('confidence', 0.6))
        
        return result
    
    async def _enhance_fast_path(self, result: Dict, query: str, context: Dict, trace: ProvenceTrace) -> Dict:
        """
        FAST enhancement: + Pattern Selection + Temporal Windows
        """
        print("🚀 FAST Enhancement: Pattern selection + temporal awareness")
        
        # === PATTERN SELECTION MODULE ===
        if self.pattern_selection:
            start_time = time.perf_counter()
            pattern_info = await self.pattern_selection.select_pattern(query, context, ComplexityLevel.FAST)
            duration = (time.perf_counter() - start_time) * 1000
            trace.add_protocol_call("pattern_selection", "select_pattern", duration, f"Selected pattern: {pattern_info.get('selected_pattern', 'none')}")
            trace.modules_invoked.append("pattern_selection")
            result['pattern_selection'] = pattern_info
            
            # === SYNTHESIS BRIDGE (Internal to Shuttle) ===
            await self._synthesize_patterns_basic(result, pattern_info, trace)
        
        # === TEMPORAL WINDOWS (Internal to Shuttle) ===
        temporal_context = await self._create_temporal_windows(query, context, trace)
        result['temporal_context'] = temporal_context
        
        return result
    
    async def _enhance_full_path(self, result: Dict, query: str, context: Dict, trace: ProvenceTrace) -> Dict:
        """
        FULL enhancement: + Decision Engine + Advanced Synthesis
        """
        print("🧠 FULL Enhancement: Decision engine + advanced synthesis")
        
        # === DECISION ENGINE MODULE ===
        if self.decision_engine:
            # Create decision options
            options = [
                {'tool': 'analyze', 'confidence': 0.8},
                {'tool': 'search', 'confidence': 0.7},
                {'tool': 'create', 'confidence': 0.6}
            ]
            
            start_time = time.perf_counter()
            decision = await self.decision_engine.make_decision(result.get('features', {}), context, options)
            duration = (time.perf_counter() - start_time) * 1000
            trace.add_protocol_call("decision_engine", "make_decision", duration, f"Selected: {decision.get('selected_option', 'none')}")
            trace.modules_invoked.append("decision_engine")
            result['decision'] = decision
            
            # === ADVANCED SYNTHESIS BRIDGE (Internal to Shuttle) ===
            await self._synthesize_patterns_advanced(result, decision, trace)
        
        # Enhanced WarpSpace operations
        if self.warp_space:
            start_time = time.perf_counter()
            manifold = await self.warp_space.create_manifold(result.get('features', {}), ComplexityLevel.FULL)
            duration = (time.perf_counter() - start_time) * 1000
            trace.add_protocol_call("warp_space", "create_manifold", duration, f"Created {manifold.get('manifold_type', 'unknown')} manifold")
            trace.modules_invoked.append("warp_space")
            result['warp_manifold'] = manifold
        
        return result
    
    async def _enhance_research_path(self, result: Dict, query: str, context: Dict, trace: ProvenceTrace) -> Dict:
        """
        RESEARCH enhancement: + Advanced WarpSpace + Full Tracing
        """
        print("🔬 RESEARCH Enhancement: Maximum capability deployment")
        
        # Advanced WarpSpace operations
        if self.warp_space:
            # Experimental operations
            experiments = ["topology_analysis", "manifold_exploration", "tensor_optimization"]
            start_time = time.perf_counter()
            experimental_results = await self.warp_space.experimental_operations(
                result.get('warp_manifold', {}), experiments
            )
            duration = (time.perf_counter() - start_time) * 1000
            trace.add_protocol_call("warp_space", "experimental_operations", duration, f"Ran {len(experiments)} experiments")
            result['experimental_warp'] = experimental_results
        
        # Advanced feature extraction with all scales
        if self.feature_extraction:
            start_time = time.perf_counter()
            advanced_features = await self.feature_extraction.extract_features(query, scales=[96, 192, 384, 768])
            duration = (time.perf_counter() - start_time) * 1000
            trace.add_protocol_call("feature_extraction", "extract_features", duration, "Extracted full-scale features")
            result['advanced_features'] = advanced_features
        
        # === FULL SPACETIME TRACING (Internal to Shuttle) ===
        result['full_spacetime_trace'] = await self._create_full_spacetime_trace(result, trace)
        
        return result
    
    # === SHUTTLE'S INTERNAL CREATIVE LOGIC ===
    
    async def _create_temporal_windows(self, query: str, context: Dict, trace: ProvenceTrace) -> Dict:
        """Shuttle's internal temporal window creation."""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["recent", "latest", "today", "now"]):
            window = {'type': 'recent_focus', 'duration': '1h', 'bias': 'recency'}
        elif any(term in query_lower for term in ["history", "past", "previous"]):
            window = {'type': 'historical', 'duration': '30d', 'bias': 'historical'}
        else:
            window = {'type': 'balanced', 'duration': '7d', 'bias': 'balanced'}
        
        trace.add_shuttle_event("temporal_window_created", f"Created {window['type']} temporal window", window)
        trace.temporal_contexts.append(window)
        
        return window
    
    async def _synthesize_patterns_basic(self, result: Dict, pattern_info: Dict, trace: ProvenceTrace):
        """Basic pattern synthesis (FAST level)."""
        if 'features' in result and 'selected_pattern' in pattern_info:
            synthesis = {
                'from_pattern': result['features'].get('dominant_pattern', 'base'),
                'to_pattern': pattern_info['selected_pattern'],
                'confidence': min(0.9, result.get('confidence', 0.5) + 0.2)
            }
            
            trace.add_shuttle_event("pattern_synthesis", "Basic pattern synthesis", synthesis)
            trace.synthesis_chain.append(synthesis)
            
            # Update confidence based on synthesis
            result['confidence'] = max(result.get('confidence', 0.5), synthesis['confidence'])
    
    async def _synthesize_patterns_advanced(self, result: Dict, decision: Dict, trace: ProvenceTrace):
        """Advanced pattern synthesis (FULL level)."""
        synthesis_events = []
        
        # Multi-modal synthesis
        if 'pattern_selection' in result and 'decision' in result:
            synthesis = {
                'type': 'cross_modal',
                'pattern_source': result['pattern_selection'].get('selected_pattern'),
                'decision_source': decision.get('selected_option'),
                'emergent_pattern': f"emergent_{int(time.time())}",
                'confidence': 0.85
            }
            synthesis_events.append(synthesis)
        
        # Feature-decision synthesis
        if 'features' in result:
            synthesis = {
                'type': 'feature_decision',
                'feature_confidence': result['features'].get('confidence', 0.5),
                'decision_confidence': decision.get('confidence', 0.5),
                'synthesis_confidence': 0.8
            }
            synthesis_events.append(synthesis)
        
        for synthesis in synthesis_events:
            trace.add_shuttle_event("advanced_synthesis", "Advanced pattern synthesis", synthesis)
            trace.synthesis_chain.append(synthesis)
    
    async def _execute_direct_routing(self, query: str, result: Dict, trace: ProvenceTrace) -> Dict:
        """Shuttle's internal direct routing for LITE level."""
        query_lower = query.lower()
        
        # Simple routing logic
        if any(term in query_lower for term in ["search", "find", "lookup"]):
            tool = "search"
        elif any(term in query_lower for term in ["analyze", "explain"]):
            tool = "analyze"
        elif any(term in query_lower for term in ["create", "generate", "make"]):
            tool = "create"
        else:
            tool = "respond"
        
        trace.add_shuttle_event("direct_routing", f"Routed to {tool}", {'tool': tool, 'confidence': 0.8})
        
        # Execute tool if protocol available
        if self.tool_execution:
            start_time = time.perf_counter()
            tool_result = await self.tool_execution.execute_tool(tool, {'query': query}, result)
            duration = (time.perf_counter() - start_time) * 1000
            trace.add_protocol_call("tool_execution", "execute_tool", duration, f"Executed {tool}")
            trace.modules_invoked.append("tool_execution")
            return tool_result
        else:
            return {
                'tool': tool,
                'output': f"Executed {tool} for: {query[:50]}...",
                'confidence': 0.8
            }
    
    async def _compute_spacetime_coordinates(self, result: Dict, trace: ProvenceTrace) -> Dict:
        """Compute spacetime coordinates for full provenance."""
        return {
            'semantic_dimensions': len(result.get('features', {}).get('embeddings', {})),
            'temporal_dimension': len(trace.temporal_contexts),
            'complexity_dimension': trace.complexity_level.value,
            'provenance_depth': len(trace.protocol_calls),
            'synthesis_depth': len(trace.synthesis_chain)
        }
    
    async def _create_full_spacetime_trace(self, result: Dict, trace: ProvenceTrace) -> Dict:
        """Create full spacetime trace for RESEARCH level."""
        return {
            'computational_graph': {
                'nodes': len(trace.modules_invoked),
                'edges': len(trace.protocol_calls),
                'synthesis_nodes': len(trace.synthesis_chain),
                'temporal_nodes': len(trace.temporal_contexts)
            },
            'execution_timeline': trace.shuttle_events,
            'performance_profile': {
                'protocol_call_times': [call['duration_ms'] for call in trace.protocol_calls],
                'total_modules': len(set(trace.modules_invoked))
            },
            'provenance_completeness': 1.0
        }
    
    def _initialize_routing_patterns(self) -> Dict:
        """Initialize routing patterns for the shuttle."""
        return {
            'greeting': ['hello', 'hi', 'hey'],
            'search': ['search', 'find', 'lookup', 'show'],
            'analysis': ['analyze', 'explain', 'compare', 'evaluate'],
            'creation': ['create', 'generate', 'make', 'build'],
            'research': ['research', 'investigate', 'explore', 'experiment']
        }
    
    def _initialize_complexity_thresholds(self) -> Dict:
        """Initialize complexity thresholds."""
        return {
            ComplexityLevel.LITE: {'confidence': 0.8, 'word_limit': 5},
            ComplexityLevel.FAST: {'confidence': 0.7, 'word_limit': 10},
            ComplexityLevel.FULL: {'confidence': 0.6, 'word_limit': 20},
            ComplexityLevel.RESEARCH: {'confidence': 0.0, 'word_limit': float('inf')}
        }
    
    # === RECURSIVE GATED MULTIPASS MEMORY CRAWLING ===
    
    async def _multipass_memory_crawl(self, query: str, complexity_level: ComplexityLevel, trace: ProvenceTrace) -> Dict:
        """
        Shuttle's internal recursive gated multipass memory crawling system.
        
        Features:
        1. Gated Retrieval - Initial retrieval at threshold T, expand high-importance results
        2. Matryoshka Importance Gating - Increasing thresholds by depth (0.6 → 0.75 → 0.85)
        3. Graph Traversal - Follow entity relationships, expand context subgraphs
        4. Multipass Fusion - Combine results from multiple passes with score fusion
        
        Benefits:
        - Richer context for complex queries
        - Better multi-hop reasoning
        - Balanced exploration/precision
        - Scalable to large graphs
        """
        if not self.memory_backend:
            trace.add_shuttle_event("multipass_crawl", "No memory backend available", {})
            return {'results': [], 'crawl_depth': 0, 'total_items': 0}
        
        # Configure crawling based on complexity level
        crawl_config = self._get_crawl_config(complexity_level)
        trace.add_shuttle_event("multipass_crawl_start", f"Starting multipass crawl", crawl_config)
        
        all_results = []
        visited_ids = set()
        crawl_stats = {
            'passes': 0,
            'total_items': 0,
            'depth_stats': {},
            'fusion_events': 0
        }
        
        # Pass 0: Initial broad exploration (threshold 0.6)
        start_time = time.perf_counter()
        initial_results = await self.memory_backend.retrieve_with_threshold(
            query, 
            threshold=crawl_config['thresholds'][0],
            limit=crawl_config['initial_limit']
        )
        
        duration = (time.perf_counter() - start_time) * 1000
        trace.add_protocol_call("memory_backend", "retrieve_with_threshold", duration, 
                               f"Pass 0: Retrieved {len(initial_results)} items at threshold {crawl_config['thresholds'][0]}")
        
        # Process initial results
        pass_results = self._process_crawl_pass(initial_results, 0, visited_ids, crawl_config)
        all_results.extend(pass_results['items'])
        crawl_stats['depth_stats'][0] = len(pass_results['items'])
        crawl_stats['total_items'] += len(pass_results['items'])
        crawl_stats['passes'] = 1
        
        # Recursive passes based on importance and complexity
        current_depth = 1
        high_importance_items = pass_results['high_importance']
        
        while (current_depth < crawl_config['max_depth'] and 
               high_importance_items and 
               len(all_results) < crawl_config['max_total_items']):
            
            pass_results = await self._execute_crawl_pass(
                high_importance_items, 
                current_depth, 
                crawl_config,
                visited_ids,
                trace
            )
            
            if pass_results['items']:
                all_results.extend(pass_results['items'])
                crawl_stats['depth_stats'][current_depth] = len(pass_results['items'])
                crawl_stats['total_items'] += len(pass_results['items'])
                crawl_stats['passes'] += 1
                
                # Prepare for next depth
                high_importance_items = pass_results['high_importance']
                current_depth += 1
            else:
                break
        
        # Multipass fusion - combine and rank results
        fused_results = await self._fuse_multipass_results(all_results, query, trace)
        crawl_stats['fusion_events'] = len(fused_results.get('fusion_events', []))
        
        trace.add_shuttle_event("multipass_crawl_complete", "Multipass crawl completed", crawl_stats)
        
        return {
            'results': fused_results['ranked_results'],
            'crawl_depth': current_depth,
            'total_items': crawl_stats['total_items'],
            'crawl_stats': crawl_stats,
            'fusion_score': fused_results.get('fusion_confidence', 0.7)
        }
    
    def _get_crawl_config(self, complexity_level: ComplexityLevel) -> Dict:
        """Get crawling configuration based on complexity level."""
        configs = {
            ComplexityLevel.LITE: {
                'max_depth': 1,
                'thresholds': [0.7],  # Single pass, high threshold
                'initial_limit': 5,
                'max_total_items': 10,
                'importance_threshold': 0.8
            },
            ComplexityLevel.FAST: {
                'max_depth': 2,
                'thresholds': [0.6, 0.75],  # Two passes
                'initial_limit': 8,
                'max_total_items': 20,
                'importance_threshold': 0.7
            },
            ComplexityLevel.FULL: {
                'max_depth': 3,
                'thresholds': [0.6, 0.75, 0.85],  # Three passes - Matryoshka gating
                'initial_limit': 12,
                'max_total_items': 35,
                'importance_threshold': 0.6
            },
            ComplexityLevel.RESEARCH: {
                'max_depth': 4,
                'thresholds': [0.5, 0.65, 0.8, 0.9],  # Deep exploration
                'initial_limit': 20,
                'max_total_items': 50,
                'importance_threshold': 0.5
            }
        }
        return configs[complexity_level]
    
    def _process_crawl_pass(self, results: List[Dict], depth: int, visited_ids: set, config: Dict) -> Dict:
        """Process results from a crawl pass."""
        processed_items = []
        high_importance = []
        
        for item in results:
            item_id = item.get('id', f"item_{hash(str(item))}")
            
            if item_id not in visited_ids:
                visited_ids.add(item_id)
                
                # Add depth and pass information
                enhanced_item = {
                    **item,
                    'crawl_depth': depth,
                    'importance_score': item.get('relevance', 0.5)
                }
                
                processed_items.append(enhanced_item)
                
                # Check if item warrants deeper exploration
                if enhanced_item['importance_score'] >= config['importance_threshold']:
                    high_importance.append(enhanced_item)
        
        return {
            'items': processed_items,
            'high_importance': high_importance
        }
    
    async def _execute_crawl_pass(self, seed_items: List[Dict], depth: int, config: Dict, 
                                 visited_ids: set, trace: ProvenceTrace) -> Dict:
        """Execute a single crawl pass."""
        if depth >= len(config['thresholds']):
            return {'items': [], 'high_importance': []}
        
        threshold = config['thresholds'][depth]
        all_pass_results = []
        
        # For each high-importance item, get related items
        for seed_item in seed_items:
            if hasattr(self.memory_backend, 'get_related'):
                start_time = time.perf_counter()
                related_items = await self.memory_backend.get_related(
                    seed_item.get('id', ''), 
                    limit=min(5, config['max_total_items'] // len(seed_items))
                )
                duration = (time.perf_counter() - start_time) * 1000
                trace.add_protocol_call("memory_backend", "get_related", duration,
                                      f"Pass {depth}: Got {len(related_items)} related items")
                
                # Filter by threshold
                filtered_items = [
                    item for item in related_items 
                    if item.get('relevance', 0.5) >= threshold
                ]
                all_pass_results.extend(filtered_items)
        
        # Process this pass
        return self._process_crawl_pass(all_pass_results, depth, visited_ids, config)
    
    async def _fuse_multipass_results(self, all_results: List[Dict], query: str, trace: ProvenceTrace) -> Dict:
        """
        Multipass fusion - combine results from multiple passes with intelligent ranking.
        
        Fusion strategies:
        1. Score fusion across depths (deeper = more focused)
        2. Deduplication by content similarity
        3. Composite scoring (relevance + importance + depth_weight)
        4. Temporal ordering consideration
        """
        if not all_results:
            return {'ranked_results': [], 'fusion_confidence': 0.0}
        
        # Deduplicate by content similarity
        deduplicated = self._deduplicate_results(all_results)
        fusion_events = [{'type': 'deduplication', 'before': len(all_results), 'after': len(deduplicated)}]
        
        # Apply composite scoring
        scored_results = []
        for item in deduplicated:
            depth = item.get('crawl_depth', 0)
            relevance = item.get('relevance', 0.5)
            importance = item.get('importance_score', 0.5)
            
            # Depth weighting: deeper results get slight boost for specificity
            depth_weight = 1.0 + (depth * 0.1)
            
            # Composite score
            composite_score = (relevance * 0.5 + importance * 0.3 + depth_weight * 0.2)
            
            scored_results.append({
                **item,
                'composite_score': composite_score,
                'fusion_depth': depth
            })
        
        # Sort by composite score
        ranked_results = sorted(scored_results, key=lambda x: x['composite_score'], reverse=True)
        
        # Calculate fusion confidence
        avg_score = sum(item['composite_score'] for item in ranked_results) / len(ranked_results)
        depth_diversity = len(set(item.get('crawl_depth', 0) for item in ranked_results))
        fusion_confidence = min(0.95, avg_score * 0.7 + depth_diversity * 0.1)
        
        fusion_events.append({
            'type': 'composite_scoring',
            'avg_score': avg_score,
            'depth_diversity': depth_diversity,
            'fusion_confidence': fusion_confidence
        })
        
        trace.add_shuttle_event("multipass_fusion", "Completed multipass result fusion", {
            'total_results': len(ranked_results),
            'fusion_confidence': fusion_confidence,
            'depth_diversity': depth_diversity
        })
        
        return {
            'ranked_results': ranked_results,
            'fusion_confidence': fusion_confidence,
            'fusion_events': fusion_events
        }
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Deduplicate results by content similarity."""
        # Simple deduplication by ID and content hash
        seen = set()
        deduplicated = []
        
        for item in results:
            # Create a simple content hash
            content = str(item.get('content', '')) + str(item.get('title', ''))
            content_hash = hash(content)
            item_id = item.get('id', str(content_hash))
            
            identifier = (item_id, content_hash)
            if identifier not in seen:
                seen.add(identifier)
                deduplicated.append(item)
        
        return deduplicated


# === DEMO PROTOCOL IMPLEMENTATIONS ===

class DemoPatternSelection:
    """Demo implementation of PatternSelectionProtocol."""
    
    async def select_pattern(self, query: str, context: Dict, complexity: ComplexityLevel) -> Dict:
        return {
            'selected_pattern': f'{complexity.name.lower()}_pattern',
            'confidence': 0.8,
            'reasoning': f'Pattern selected for {complexity.name} complexity level'
        }
    
    async def assess_pattern_necessity(self, query: str) -> float:
        return 0.7 if len(query.split()) > 5 else 0.3
    
    async def synthesize_patterns(self, primary: Dict, secondary: Dict) -> Dict:
        return {
            'synthesized_pattern': f"synthesis_{primary.get('type', 'unknown')}_{secondary.get('type', 'unknown')}",
            'confidence': 0.85
        }


class DemoDecisionEngine:
    """Demo implementation of DecisionEngineProtocol."""
    
    async def make_decision(self, features: Dict, context: Dict, options: List[Dict]) -> Dict:
        # Simple decision: pick highest confidence option
        best_option = max(options, key=lambda x: x.get('confidence', 0))
        return {
            'selected_option': best_option.get('tool', 'respond'),
            'confidence': best_option.get('confidence', 0.5),
            'reasoning': 'Selected highest confidence option'
        }
    
    async def assess_decision_complexity(self, features: Dict) -> float:
        return features.get('confidence', 0.5)
    
    async def optimize_multi_criteria(self, criteria: List[Dict], constraints: Dict) -> Dict:
        return {
            'optimized_solution': 'pareto_optimal',
            'criteria_weights': [1.0 / len(criteria)] * len(criteria),
            'satisfaction_score': 0.85
        }


class DemoToolExecution:
    """Demo implementation of ToolExecutionProtocol."""
    
    async def execute_tool(self, tool_name: str, parameters: Dict, context: Dict) -> Dict:
        return {
            'tool': tool_name,
            'output': f"Executed {tool_name} with query: {parameters.get('query', 'unknown')}",
            'confidence': 0.8,
            'execution_time_ms': 10
        }
    
    async def list_available_tools(self, context: Dict) -> List[Dict]:
        return [
            {'name': 'search', 'description': 'Search for information'},
            {'name': 'analyze', 'description': 'Analyze data'},
            {'name': 'create', 'description': 'Create content'},
            {'name': 'respond', 'description': 'Generate response'}
        ]
    
    async def assess_tool_necessity(self, query: str, context: Dict) -> Dict:
        return {
            'recommended_tools': ['search', 'analyze'],
            'confidence': 0.7
        }


class DemoMemoryBackend:
    """Demo implementation of MemoryBackendProtocol with multipass crawling support."""
    
    def __init__(self):
        # Simulate a knowledge graph for demo
        self.knowledge_graph = {
            'bee_1': {
                'id': 'bee_1',
                'content': 'Honeybee colony behavior patterns',
                'relevance': 0.9,
                'related': ['hive_1', 'environment_1']
            },
            'hive_1': {
                'id': 'hive_1', 
                'content': 'Hive health monitoring systems',
                'relevance': 0.85,
                'related': ['bee_1', 'disease_1']
            },
            'environment_1': {
                'id': 'environment_1',
                'content': 'Environmental factors affecting bee colonies',
                'relevance': 0.8,
                'related': ['bee_1', 'climate_1']
            },
            'disease_1': {
                'id': 'disease_1',
                'content': 'Bee disease detection and prevention',
                'relevance': 0.75,
                'related': ['hive_1', 'treatment_1']
            },
            'climate_1': {
                'id': 'climate_1',
                'content': 'Climate change impact on bee populations',
                'relevance': 0.7,
                'related': ['environment_1']
            },
            'treatment_1': {
                'id': 'treatment_1',
                'content': 'Advanced bee disease treatment protocols',
                'relevance': 0.65,
                'related': ['disease_1']
            }
        }
    
    async def store(self, data: Dict, metadata: Optional[Dict] = None) -> str:
        item_id = f"item_{len(self.knowledge_graph)}"
        self.knowledge_graph[item_id] = {**data, 'id': item_id}
        return item_id
    
    async def retrieve(self, query: str, filters: Optional[Dict] = None, limit: int = 10) -> List[Dict]:
        # Simple keyword matching for demo
        query_lower = query.lower()
        results = []
        
        for item in self.knowledge_graph.values():
            if any(word in item['content'].lower() for word in query_lower.split()):
                results.append(item)
        
        return sorted(results, key=lambda x: x['relevance'], reverse=True)[:limit]
    
    async def retrieve_with_threshold(self, query: str, threshold: float, filters: Optional[Dict] = None, limit: int = 10) -> List[Dict]:
        """Retrieve data with specific similarity/relevance threshold."""
        base_results = await self.retrieve(query, filters, limit * 2)  # Get more to filter
        filtered_results = [item for item in base_results if item['relevance'] >= threshold]
        return filtered_results[:limit]
    
    async def get_related(self, item_id: str, relationship_types: Optional[List[str]] = None, limit: int = 10) -> List[Dict]:
        """Get items related to given item (for graph traversal)."""
        if item_id not in self.knowledge_graph:
            return []
        
        related_ids = self.knowledge_graph[item_id].get('related', [])
        related_items = []
        
        for rel_id in related_ids[:limit]:
            if rel_id in self.knowledge_graph:
                related_items.append(self.knowledge_graph[rel_id])
        
        return related_items
    
    async def get_context_subgraph(self, item_ids: List[str], depth: int = 1) -> Dict:
        """Get context subgraph around given items."""
        subgraph = {}
        visited = set()
        
        async def expand_node(node_id: str, current_depth: int):
            if current_depth > depth or node_id in visited or node_id not in self.knowledge_graph:
                return
            
            visited.add(node_id)
            subgraph[node_id] = self.knowledge_graph[node_id]
            
            if current_depth < depth:
                for related_id in self.knowledge_graph[node_id].get('related', []):
                    await expand_node(related_id, current_depth + 1)
        
        for item_id in item_ids:
            await expand_node(item_id, 0)
        
        return {'nodes': subgraph, 'total_nodes': len(subgraph)}
    
    async def get_by_id(self, item_id: str) -> Optional[Dict]:
        return self.knowledge_graph.get(item_id)
    
    async def update(self, item_id: str, data: Dict) -> bool:
        if item_id in self.knowledge_graph:
            self.knowledge_graph[item_id].update(data)
            return True
        return False
    
    async def delete(self, item_id: str) -> bool:
        if item_id in self.knowledge_graph:
            del self.knowledge_graph[item_id]
            return True
        return False
    
    async def health_check(self) -> bool:
        return True


async def demo_protocol_architecture():
    """Demonstrate the complete Protocol + Modules = mythRL architecture."""
    
    print("🚀 PROTOCOL + MODULES = mythRL ARCHITECTURE DEMO")
    print("=" * 70)
    print("Core Philosophy:")
    print("• Shuttle = Creative Orchestrator with Internal Intelligence")
    print("• Protocols = Swappable Interface Contracts")
    print("• Modules = Domain-Specific Implementations")
    print("• 3-5-7-9 = Progressive Complexity Activation")
    print("• Full Computational Provenance via Spacetime Tracing")
    print()
    
    # Create MythRL Shuttle
    shuttle = MythRLShuttle()
    
    # Register protocol implementations
    shuttle.register_protocol('pattern_selection', DemoPatternSelection())
    shuttle.register_protocol('decision_engine', DemoDecisionEngine())
    shuttle.register_protocol('memory_backend', DemoMemoryBackend())
    shuttle.register_protocol('tool_execution', DemoToolExecution())
    
    # Test queries with different complexity levels
    test_queries = [
        "hello there",  # LITE
        "search for beekeeping patterns in the database",  # FAST  
        "analyze the complex relationship between hive health and environmental factors",  # FULL
        "research innovative multi-dimensional optimization strategies for advanced colony management systems"  # RESEARCH
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {query}")
        print('='*70)
        
        result = await shuttle.weave(query)
        
        print(f"\n📊 MYTHRL RESULT:")
        print(f"  Query: {result.query}")
        print(f"  Complexity Level: {result.complexity_level.name} ({result.complexity_level.value} steps)")
        print(f"  Final Confidence: {result.confidence:.2f}")
        print(f"  Output: {result.output[:60]}...")
        
        perf = result.get_performance_summary()
        print(f"\n⚡ PERFORMANCE:")
        print(f"  Duration: {perf['total_duration_ms']:.1f}ms")
        print(f"  Modules Used: {perf['modules_used']}")
        print(f"  Protocol Calls: {perf['protocol_calls']}")
        print(f"  Shuttle Events: {perf['shuttle_events']}")
        
        print(f"\n🧬 PROVENANCE:")
        print(f"  Modules Invoked: {result.provenance.modules_invoked}")
        print(f"  Synthesis Events: {len(result.provenance.synthesis_chain)}")
        print(f"  Temporal Contexts: {len(result.provenance.temporal_contexts)}")
        
        print(f"\n🌌 SPACETIME COORDINATES:")
        for coord, value in result.spacetime_coordinates.items():
            print(f"  {coord}: {value}")


if __name__ == "__main__":
    asyncio.run(demo_protocol_architecture())