#!/usr/bin/env python3
"""
Token-Aware MythRL Shuttle with Bayesian Budget Management
===========================================================
Enhanced Shuttle architecture that includes intelligent token budgeting
with Bayesian resource allocation throughout all protocol operations.

Key Features:
- Complexity-aware token allocation
- Protocol-specific budget tracking
- Bayesian usage prediction
- Dynamic budget reallocation
- Token estimation for all operations
- Budget enforcement with graceful degradation
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

# Import base components
try:
    from dev.protocol_modules_mythrl import (
        ComplexityLevel, ProvenceTrace, MythRLResult, MythRLShuttle,
        PatternSelectionProtocol, DecisionEngineProtocol, MemoryBackendProtocol,
        FeatureExtractionProtocol, WarpSpaceProtocol, ToolExecutionProtocol
    )
    from bayesian_symphony_protocols import BayesianTokenBudgetManager, TokenBudget
except ImportError:
    # Fallback definitions for standalone operation
    class ComplexityLevel(Enum):
        LITE = 3
        FAST = 5
        FULL = 7
        RESEARCH = 9
    
    @dataclass
    class ProvenceTrace:
        operation_id: str
        complexity_level: ComplexityLevel
        modules_invoked: List[str] = field(default_factory=list)
        protocol_calls: List[Dict] = field(default_factory=list)
        token_usage: Dict[str, int] = field(default_factory=dict)
        budget_allocations: List[Dict] = field(default_factory=list)
    
    @dataclass
    class MythRLResult:
        response: str
        confidence: float
        complexity_level: ComplexityLevel
        provenance: ProvenceTrace
        token_budget: Optional[Dict] = None
        
    @dataclass 
    class TokenBudget:
        total_budget: int
        allocated: Dict[str, int] = field(default_factory=dict)
        used: Dict[str, int] = field(default_factory=dict)
        uncertainty_buffer: int = 0


# ============================================================================
# TOKEN-AWARE PROTOCOL INTERFACES
# ============================================================================

class TokenAwareProtocol:
    """Base class for protocols that track token usage."""
    
    def __init__(self):
        self.total_tokens_used = 0
        self.operation_count = 0
        
    def estimate_tokens(self, operation: str, data_size: int, complexity: ComplexityLevel) -> int:
        """Estimate token usage for an operation."""
        base_tokens = {
            'select_pattern': 30,
            'extract_features': 80,
            'retrieve': 150,
            'make_decision': 60,
            'tensor_operation': 100,
            'execute_tool': 200
        }.get(operation, 50)
        
        # Complexity multiplier
        multiplier = {
            ComplexityLevel.LITE: 0.5,
            ComplexityLevel.FAST: 1.0,
            ComplexityLevel.FULL: 1.8,
            ComplexityLevel.RESEARCH: 3.2
        }[complexity]
        
        # Data size factor (tokens scale with input size)
        size_factor = max(1.0, np.log10(max(1, data_size)) / 2)
        
        return int(base_tokens * multiplier * size_factor)
    
    def record_usage(self, tokens_used: int):
        """Record actual token usage."""
        self.total_tokens_used += tokens_used
        self.operation_count += 1
    
    def get_average_usage(self) -> float:
        """Get average tokens per operation."""
        return self.total_tokens_used / max(1, self.operation_count)


# ============================================================================
# TOKEN-AWARE SHUTTLE IMPLEMENTATION
# ============================================================================

class TokenAwareMythRLShuttle:
    """
    MythRL Shuttle with integrated Bayesian token budget management.
    
    Features:
    - Pre-flight budget allocation
    - Real-time usage tracking
    - Dynamic budget reallocation
    - Graceful degradation when budget is exceeded
    - Learning from usage patterns
    """
    
    def __init__(self, base_token_budget: int = 8000):
        self.base_token_budget = base_token_budget
        
        # Bayesian budget manager
        self.budget_manager = BayesianTokenBudgetManager(base_token_budget)
        
        # Registered protocols
        self.protocols: Dict[str, Any] = {}
        
        # Token tracking
        self.current_budget: Optional[TokenBudget] = None
        self.operation_count = 0
        
        # Performance metrics
        self.budget_accuracy_history = []
        self.protocol_efficiency = {}
    
    def register_protocol(self, name: str, implementation: Any):
        """Register a protocol implementation."""
        self.protocols[name] = implementation
        
        # Initialize efficiency tracking
        self.protocol_efficiency[name] = {
            'estimated_total': 0,
            'actual_total': 0,
            'operations': 0
        }
    
    async def weave(
        self, 
        query: str, 
        context: Optional[Dict] = None,
        max_tokens: Optional[int] = None
    ) -> MythRLResult:
        """
        Main weaving operation with token budget management.
        
        Args:
            query: Input query
            context: Optional context
            max_tokens: Override default budget
            
        Returns:
            MythRLResult with budget information
        """
        start_time = time.perf_counter()
        
        # Create operation trace
        trace = ProvenceTrace(
            operation_id=f"weave_{self.operation_count}",
            complexity_level=ComplexityLevel.FAST  # Will be updated
        )
        
        # Step 1: Assess complexity and create budget
        complexity = await self._assess_complexity_with_tokens(query, context)
        trace.complexity_level = complexity
        
        # Step 2: Create token budget
        context_tokens = self._estimate_context_tokens(context) if context else 0
        budget_override = max_tokens or self.base_token_budget
        
        self.current_budget = self.budget_manager.create_complexity_budget(
            complexity, 
            context_tokens
        )
        
        # Override total budget if specified
        if max_tokens:
            self.current_budget.total_budget = max_tokens
        
        trace.budget_allocations.append({
            'phase': 'initial_allocation',
            'total_budget': self.current_budget.total_budget,
            'allocated': dict(self.current_budget.allocated),
            'uncertainty_buffer': self.current_budget.uncertainty_buffer
        })
        
        # Step 3: Execute weaving with budget enforcement
        try:
            result = await self._execute_weaving_with_budget(query, context, complexity, trace)
            
            # Step 4: Update budget learning
            self._update_budget_learning(trace)
            
            return result
            
        except TokenBudgetExceeded as e:
            # Graceful degradation
            return await self._handle_budget_exceeded(query, context, complexity, trace, e)
        
        finally:
            self.operation_count += 1
    
    async def _assess_complexity_with_tokens(
        self, 
        query: str, 
        context: Optional[Dict]
    ) -> ComplexityLevel:
        """Assess complexity considering token implications."""
        
        # Basic complexity indicators
        query_length = len(query.split())
        context_size = len(str(context)) if context else 0
        
        # Simple heuristic with token awareness
        if query_length <= 10 and context_size <= 500:
            return ComplexityLevel.LITE
        elif query_length <= 30 and context_size <= 2000:
            return ComplexityLevel.FAST
        elif query_length <= 100 and context_size <= 8000:
            return ComplexityLevel.FULL
        else:
            return ComplexityLevel.RESEARCH
    
    def _estimate_context_tokens(self, context: Dict) -> int:
        """Estimate tokens needed for context."""
        if not context:
            return 0
        
        # Rough estimation: 1 token per 4 characters
        context_str = str(context)
        return len(context_str) // 4
    
    async def _execute_weaving_with_budget(
        self, 
        query: str, 
        context: Optional[Dict],
        complexity: ComplexityLevel,
        trace: ProvenceTrace
    ) -> MythRLResult:
        """Execute weaving process with token budget enforcement."""
        
        result = {'query': query, 'response': '', 'features': {}}
        
        # Phase 1: Pattern Selection (if complexity >= FAST)
        if complexity.value >= ComplexityLevel.FAST.value:
            result = await self._execute_protocol_with_budget(
                'pattern_selection', 
                'select_pattern',
                {'query': query, 'context': context, 'complexity': complexity},
                trace
            )
        
        # Phase 2: Feature Extraction
        result = await self._execute_protocol_with_budget(
            'feature_extraction',
            'extract_features', 
            {'query': query, 'result': result, 'complexity': complexity},
            trace
        )
        
        # Phase 3: Memory Retrieval
        if 'memory_backend' in self.protocols:
            memory_result = await self._execute_protocol_with_budget(
                'memory_backend',
                'retrieve',
                {'query': query, 'threshold': 0.7, 'limit': 10, 'complexity': complexity},
                trace
            )
            result['memory'] = memory_result
        
        # Phase 4: Decision Engine (if complexity >= FULL)
        if complexity.value >= ComplexityLevel.FULL.value:
            decision_result = await self._execute_protocol_with_budget(
                'decision_engine',
                'make_decision',
                {'query': query, 'features': result['features'], 'complexity': complexity},
                trace
            )
            result['decision'] = decision_result
        
        # Phase 5: WarpSpace (if complexity >= RESEARCH)
        if complexity.value >= ComplexityLevel.RESEARCH.value:
            warp_result = await self._execute_protocol_with_budget(
                'warpspace',
                'tensor_operation',
                {'data': result, 'complexity': complexity, 'operation_type': 'advanced'},
                trace
            )
            result['warp'] = warp_result
        
        # Phase 6: Tool Execution
        tool_result = await self._execute_protocol_with_budget(
            'tool_execution',
            'execute_tool',
            {'tool_name': 'respond', 'parameters': {'query': query}, 'context': result},
            trace
        )
        result['tool_output'] = tool_result
        
        # Generate final response
        confidence = result.get('features', {}).get('confidence', 0.7)
        response = tool_result.get('output', f"Processed query: {query}")
        
        return MythRLResult(
            response=response,
            confidence=confidence,
            complexity_level=complexity,
            provenance=trace,
            token_budget={
                'total_budget': self.current_budget.total_budget,
                'total_used': sum(self.current_budget.used.values()),
                'remaining': self.current_budget.remaining(),
                'allocation_accuracy': self._compute_allocation_accuracy()
            }
        )
    
    async def _execute_protocol_with_budget(
        self,
        protocol_name: str,
        method_name: str,
        parameters: Dict,
        trace: ProvenceTrace
    ) -> Dict:
        """Execute protocol method with budget tracking."""
        
        if protocol_name not in self.protocols:
            # Mock execution for missing protocols
            return await self._mock_protocol_execution(protocol_name, method_name, parameters)
        
        # Check budget availability
        allocated_tokens = self.current_budget.allocated.get(protocol_name, 0)
        used_tokens = self.current_budget.used.get(protocol_name, 0)
        available_tokens = allocated_tokens - used_tokens
        
        if available_tokens <= 0:
            # Try to reallocate budget
            reallocation_success = await self._attempt_budget_reallocation(protocol_name)
            if not reallocation_success:
                raise TokenBudgetExceeded(f"No budget available for {protocol_name}")
        
        # Estimate tokens for this operation
        data_size = len(str(parameters))
        estimated_tokens = self._estimate_operation_tokens(
            protocol_name, 
            method_name, 
            data_size,
            parameters.get('complexity', ComplexityLevel.FAST)
        )
        
        # Execute protocol
        start_time = time.perf_counter()
        
        try:
            protocol = self.protocols[protocol_name]
            
            # Call appropriate method
            if hasattr(protocol, method_name):
                if method_name == 'select_pattern':
                    result = await protocol.select_pattern(
                        parameters['query'],
                        parameters.get('context', {}),
                        parameters['complexity']
                    )
                elif method_name == 'extract_features':
                    result = await protocol.extract_features(
                        parameters['query'],
                        parameters.get('result', {}),
                        parameters['complexity']
                    )
                elif method_name == 'retrieve':
                    result = await protocol.retrieve(
                        parameters['query'],
                        parameters.get('threshold', 0.7),
                        parameters.get('limit', 10),
                        parameters['complexity']
                    )
                elif method_name == 'make_decision':
                    result = await protocol.make_decision(
                        parameters.get('features', {}),
                        parameters.get('context', {}),
                        parameters['complexity']
                    )
                elif method_name == 'tensor_operation':
                    result = await protocol.tensor_operation(
                        parameters['data'],
                        parameters['complexity'],
                        parameters.get('operation_type', 'standard')
                    )
                elif method_name == 'execute_tool':
                    result = await protocol.execute_tool(
                        parameters['tool_name'],
                        parameters['parameters'],
                        parameters['context']
                    )
                else:
                    result = {'status': 'unknown_method', 'method': method_name}
            else:
                result = {'status': 'method_not_found', 'method': method_name}
        
        except Exception as e:
            result = {'status': 'error', 'error': str(e)}
        
        # Calculate actual token usage (simulated)
        duration_ms = (time.perf_counter() - start_time) * 1000
        actual_tokens = max(1, int(estimated_tokens * (0.8 + 0.4 * np.random.random())))
        
        # Update budget tracking
        if protocol_name not in self.current_budget.used:
            self.current_budget.used[protocol_name] = 0
        self.current_budget.used[protocol_name] += actual_tokens
        
        # Update learning
        self.budget_manager.update_usage(protocol_name, actual_tokens)
        
        # Update efficiency tracking
        if protocol_name in self.protocol_efficiency:
            eff = self.protocol_efficiency[protocol_name]
            eff['estimated_total'] += estimated_tokens
            eff['actual_total'] += actual_tokens
            eff['operations'] += 1
        
        # Record in trace
        trace.protocol_calls.append({
            'protocol': protocol_name,
            'method': method_name,
            'estimated_tokens': estimated_tokens,
            'actual_tokens': actual_tokens,
            'duration_ms': duration_ms,
            'result_summary': str(result)[:100]
        })
        
        if protocol_name not in trace.token_usage:
            trace.token_usage[protocol_name] = 0
        trace.token_usage[protocol_name] += actual_tokens
        
        trace.modules_invoked.append(protocol_name)
        
        return result
    
    async def _mock_protocol_execution(
        self, 
        protocol_name: str, 
        method_name: str, 
        parameters: Dict
    ) -> Dict:
        """Mock protocol execution for missing implementations."""
        
        # Simulate realistic execution time
        await asyncio.sleep(0.01 + 0.02 * np.random.random())
        
        return {
            'protocol': protocol_name,
            'method': method_name,
            'status': 'mocked',
            'confidence': 0.6 + 0.3 * np.random.random(),
            'result': f"Mock result for {protocol_name}.{method_name}"
        }
    
    def _estimate_operation_tokens(
        self,
        protocol_name: str,
        method_name: str,
        data_size: int,
        complexity: ComplexityLevel
    ) -> int:
        """Estimate tokens for a specific operation."""
        
        # Base estimates per protocol/method
        base_estimates = {
            'pattern_selection': {'select_pattern': 40},
            'feature_extraction': {'extract_features': 90},
            'memory_backend': {'retrieve': 180},
            'decision_engine': {'make_decision': 70},
            'warpspace': {'tensor_operation': 120},
            'tool_execution': {'execute_tool': 250}
        }
        
        base_tokens = base_estimates.get(protocol_name, {}).get(method_name, 60)
        
        # Complexity multiplier
        multiplier = {
            ComplexityLevel.LITE: 0.6,
            ComplexityLevel.FAST: 1.0,
            ComplexityLevel.FULL: 1.7,
            ComplexityLevel.RESEARCH: 3.0
        }[complexity]
        
        # Data size factor
        size_factor = max(1.0, np.log10(max(1, data_size)) / 3)
        
        return int(base_tokens * multiplier * size_factor)
    
    async def _attempt_budget_reallocation(self, protocol_name: str) -> bool:
        """Attempt to reallocate budget from other protocols."""
        
        # Find protocols with unused budget
        available_protocols = []
        for pname, allocated in self.current_budget.allocated.items():
            used = self.current_budget.used.get(pname, 0)
            unused = allocated - used
            if unused > 50:  # Minimum reallocation threshold
                available_protocols.append((pname, unused))
        
        if not available_protocols:
            return False
        
        # Reallocate from protocol with most unused budget
        source_protocol, unused_amount = max(available_protocols, key=lambda x: x[1])
        reallocation_amount = min(unused_amount // 2, 200)  # Reallocate up to half
        
        # Update allocations
        self.current_budget.allocated[source_protocol] -= reallocation_amount
        if protocol_name not in self.current_budget.allocated:
            self.current_budget.allocated[protocol_name] = 0
        self.current_budget.allocated[protocol_name] += reallocation_amount
        
        # Record reallocation
        self.current_budget.allocated['_reallocations'] = self.current_budget.allocated.get('_reallocations', 0) + 1
        
        return True
    
    def _compute_allocation_accuracy(self) -> float:
        """Compute accuracy of budget allocation."""
        if not self.current_budget:
            return 0.0
        
        total_allocated = sum(self.current_budget.allocated.values())
        total_used = sum(self.current_budget.used.values())
        
        if total_allocated == 0:
            return 0.0
        
        # Accuracy is how close actual usage is to allocation
        accuracy = 1.0 - abs(total_used - total_allocated) / total_allocated
        return max(0.0, min(1.0, accuracy))
    
    def _update_budget_learning(self, trace: ProvenceTrace):
        """Update budget learning from completed operation."""
        
        # Record allocation accuracy
        accuracy = self._compute_allocation_accuracy()
        self.budget_accuracy_history.append(accuracy)
        
        # Keep only recent history
        if len(self.budget_accuracy_history) > 100:
            self.budget_accuracy_history = self.budget_accuracy_history[-100:]
    
    async def _handle_budget_exceeded(
        self,
        query: str,
        context: Optional[Dict],
        complexity: ComplexityLevel,
        trace: ProvenceTrace,
        exception: 'TokenBudgetExceeded'
    ) -> MythRLResult:
        """Handle budget exceeded with graceful degradation."""
        
        # Reduce complexity and try again
        if complexity.value > ComplexityLevel.LITE.value:
            degraded_complexity = ComplexityLevel(complexity.value - 2)
            
            # Create smaller budget
            self.current_budget = self.budget_manager.create_complexity_budget(
                degraded_complexity,
                context_size=0  # Skip context to save tokens
            )
            
            # Simplified execution
            result = await self._execute_simplified_weaving(query, degraded_complexity, trace)
            
            return MythRLResult(
                response=f"[Budget Limited] {result.get('response', 'Simplified response')}",
                confidence=0.4,  # Lower confidence due to degradation
                complexity_level=degraded_complexity,
                provenance=trace,
                token_budget={
                    'budget_exceeded': True,
                    'original_complexity': complexity.name,
                    'degraded_complexity': degraded_complexity.name,
                    'exception': str(exception)
                }
            )
        else:
            # Already at minimum complexity
            return MythRLResult(
                response=f"[Budget Exceeded] Unable to process query within token limits.",
                confidence=0.1,
                complexity_level=complexity,
                provenance=trace,
                token_budget={'budget_exceeded': True, 'exception': str(exception)}
            )
    
    async def _execute_simplified_weaving(
        self,
        query: str,
        complexity: ComplexityLevel,
        trace: ProvenceTrace
    ) -> Dict:
        """Execute simplified weaving when budget is constrained."""
        
        # Only essential operations
        result = {
            'query': query,
            'response': f"Processed: {query[:50]}..." if len(query) > 50 else query,
            'simplified': True
        }
        
        # Minimal feature extraction
        if 'feature_extraction' in self.protocols:
            try:
                features = await self._execute_protocol_with_budget(
                    'feature_extraction',
                    'extract_features',
                    {'query': query, 'result': {}, 'complexity': complexity},
                    trace
                )
                result['features'] = features
            except Exception:
                result['features'] = {'confidence': 0.5}
        
        return result
    
    def get_budget_analytics(self) -> Dict:
        """Get comprehensive budget analytics."""
        
        budget_analytics = self.budget_manager.get_budget_analytics()
        
        # Add shuttle-specific metrics
        budget_analytics.update({
            'allocation_accuracy_history': self.budget_accuracy_history[-10:],
            'average_allocation_accuracy': np.mean(self.budget_accuracy_history) if self.budget_accuracy_history else 0.0,
            'protocol_efficiency': {},
            'total_operations': self.operation_count
        })
        
        # Compute protocol efficiency
        for protocol, stats in self.protocol_efficiency.items():
            if stats['operations'] > 0:
                estimated_avg = stats['estimated_total'] / stats['operations']
                actual_avg = stats['actual_total'] / stats['operations']
                efficiency = estimated_avg / max(1, actual_avg)  # >1 means overestimate, <1 means underestimate
                
                budget_analytics['protocol_efficiency'][protocol] = {
                    'operations': stats['operations'],
                    'estimated_avg': estimated_avg,
                    'actual_avg': actual_avg,
                    'efficiency_ratio': efficiency,
                    'prediction_accuracy': 1.0 - abs(1.0 - efficiency)
                }
        
        return budget_analytics


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class TokenBudgetExceeded(Exception):
    """Exception raised when token budget is exceeded."""
    pass


# ============================================================================
# DEMONSTRATION
# ============================================================================

async def demonstrate_token_aware_shuttle():
    """Demonstrate token-aware shuttle with budget management."""
    
    print("🚀 TOKEN-AWARE MYTHRL SHUTTLE DEMONSTRATION 🚀")
    print("=" * 80)
    
    # Create shuttle with budget
    shuttle = TokenAwareMythRLShuttle(base_token_budget=5000)
    
    # Mock protocol implementations
    class MockPatternSelection:
        async def select_pattern(self, query: str, context: Dict, complexity: ComplexityLevel) -> Dict:
            await asyncio.sleep(0.01)
            return {
                'selected_pattern': 'analytical' if 'analyze' in query.lower() else 'conversational',
                'confidence': 0.8,
                'alternatives': ['factual', 'creative']
            }
    
    class MockFeatureExtraction:
        async def extract_features(self, query: str, result: Dict, complexity: ComplexityLevel) -> Dict:
            await asyncio.sleep(0.02)
            return {
                'features': [0.1, 0.5, 0.3, 0.7, 0.2],
                'confidence': 0.75,
                'extraction_method': 'bayesian_neural_network'
            }
    
    class MockMemoryBackend:
        async def retrieve(self, query: str, threshold: float, limit: int, complexity: ComplexityLevel) -> Dict:
            await asyncio.sleep(0.03)
            return {
                'shards': [
                    {'id': 'mem1', 'content': 'Relevant memory 1', 'relevance': 0.85},
                    {'id': 'mem2', 'content': 'Relevant memory 2', 'relevance': 0.72}
                ],
                'total_found': 2,
                'retrieval_method': 'hierarchical_bayesian'
            }
    
    class MockToolExecution:
        async def execute_tool(self, tool_name: str, parameters: Dict, context: Dict) -> Dict:
            await asyncio.sleep(0.02)
            return {
                'tool': tool_name,
                'output': f"Executed {tool_name}: {parameters.get('query', 'N/A')}",
                'confidence': 0.8
            }
    
    # Register protocols
    shuttle.register_protocol('pattern_selection', MockPatternSelection())
    shuttle.register_protocol('feature_extraction', MockFeatureExtraction())
    shuttle.register_protocol('memory_backend', MockMemoryBackend())
    shuttle.register_protocol('tool_execution', MockToolExecution())
    
    print("\n📊 BUDGET ALLOCATION TESTS")
    print("-" * 50)
    
    # Test 1: Simple query (LITE complexity)
    print("\n1. Simple Query (LITE complexity):")
    result1 = await shuttle.weave("Hello", max_tokens=1000)
    print(f"   Response: {result1.response[:60]}...")
    print(f"   Complexity: {result1.complexity_level.name}")
    print(f"   Tokens Used: {result1.token_budget['total_used']}/{result1.token_budget['total_budget']}")
    print(f"   Remaining: {result1.token_budget['remaining']}")
    
    # Test 2: Complex query (FULL complexity)
    print("\n2. Complex Analysis Query (FULL complexity):")
    result2 = await shuttle.weave(
        "Analyze the complex relationships between quantum mechanics and consciousness in modern neuroscience research",
        context={'previous_analysis': 'Some complex context data here'},
        max_tokens=3000
    )
    print(f"   Response: {result2.response[:60]}...")
    print(f"   Complexity: {result2.complexity_level.name}")
    print(f"   Tokens Used: {result2.token_budget['total_used']}/{result2.token_budget['total_budget']}")
    print(f"   Allocation Accuracy: {result2.token_budget['allocation_accuracy']:.3f}")
    
    # Test 3: Budget-constrained query
    print("\n3. Budget-Constrained Query:")
    result3 = await shuttle.weave(
        "This is a very long and complex query that should exceed the small token budget allocated to it and trigger graceful degradation mechanisms",
        max_tokens=200  # Very small budget
    )
    print(f"   Response: {result3.response[:60]}...")
    print(f"   Complexity: {result3.complexity_level.name}")
    if 'budget_exceeded' in result3.token_budget:
        print(f"   Budget Exceeded: {result3.token_budget['budget_exceeded']}")
        if 'degraded_complexity' in result3.token_budget:
            print(f"   Degraded to: {result3.token_budget['degraded_complexity']}")
    
    # Analytics
    print("\n📈 BUDGET ANALYTICS")
    print("-" * 50)
    
    analytics = shuttle.get_budget_analytics()
    print(f"Total Operations: {analytics['total_operations']}")
    print(f"Average Allocation Accuracy: {analytics['average_allocation_accuracy']:.3f}")
    print(f"Protocols Tracked: {analytics['total_protocols_tracked']}")
    
    print("\nProtocol Efficiency:")
    for protocol, stats in analytics['protocol_efficiency'].items():
        print(f"  {protocol}: {stats['prediction_accuracy']:.3f} accuracy "
              f"({stats['operations']} ops, {stats['actual_avg']:.1f} avg tokens)")
    
    print("\n" + "=" * 80)
    print("🎯 TOKEN BUDGETING COMPLETE!")
    print("✨ Bayesian resource allocation with graceful degradation! ✨")
    print("💰 Complexity-aware budgeting with real-time reallocation!")


if __name__ == "__main__":
    asyncio.run(demonstrate_token_aware_shuttle())