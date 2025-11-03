# Self-Tuning Strategy: Configuration Level 4 → 1

**Date**: November 2, 2025
**Goal**: Eliminate 72 configuration parameters through intelligent self-tuning
**Philosophy**: *"The system should configure itself based on what actually works."*

---

## Executive Summary

**Current State**: Configuration Nightmare (Level 4)
- 72 parameters across 9 subsystems
- 19 parameters never customized (26% waste)
- 31 dependent parameters creating brittle interdependencies
- Hardcoded timeouts (2s → 200ms) based on analysis, not measurement

**Target State**: Self-Tuning Elegance (Level 1)
- 3 high-level knobs: `mode` (BARE/FAST/FUSED), `environment` (DEV/STAGING/PROD), `memory_backend` (INMEMORY/HYBRID/HYPERSPACE)
- All other parameters learned from production metrics
- Zero-configuration for 95% of use cases
- Adaptive to hardware, load, and query patterns

**Path**: Deploy swarm of self-tuning agents, each responsible for one subsystem

---

## I. The Agent Swarm Architecture

### Core Principle: Decentralized Parameter Tuning

Instead of one monolithic tuner, deploy **7 specialized tuning agents**, each managing one configuration domain:

```
┌─────────────────────────────────────────────────────────┐
│  Master Tuning Coordinator (Thompson Sampling)          │
│  Selects which agent to activate based on impact        │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ TimeoutTuner │  │  CacheTuner  │  │ ThresholdTune│
│ (9 params)   │  │  (6 params)  │  │  (12 params) │
└──────────────┘  └──────────────┘  └──────────────┘
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  MemoryTuner │  │ ComplexityTun│  │  PolicyTuner │
│  (11 params) │  │  (8 params)  │  │  (4 params)  │
└──────────────┘  └──────────────┘  └──────────────┘
        ▼
┌──────────────┐
│  PhysicsTuner│
│  (7 params)  │
└──────────────┘
```

Each agent:
- **Measures** its subsystem's performance
- **Explores** parameter space using Thompson Sampling
- **Learns** from outcomes (latency, success rate, resource usage)
- **Persists** learned parameters to disk
- **Reports** tuning decisions to Prometheus metrics

---

## II. Agent Specifications

### Agent 1: TimeoutTuner

**Responsibility**: Adaptive timeout management
**Parameters Managed** (9):
```python
pipeline_timeout: float = 5.0
retrieval_timeout: float = 0.2
stage_timeouts: Dict[str, float] = {
    'features': 2.0,
    'retrieval': 2.0,
    'decision': 2.0,
    'execution': 3.0,
}
max_autonomous_duration: float = 3600.0
message_timeout_sec: int = 120
```

**Current Problem**:
- Hardcoded 200ms retrieval timeout (was 2s)
- Fixed stage timeouts regardless of query complexity
- No adaptation to system load or hardware speed

**Self-Tuning Strategy**:

1. **Measurement Phase** (first 100 queries):
   - Track p50, p95, p99 latencies per stage
   - Measure actual pipeline duration distribution
   - Collect system load metrics (CPU%, memory%)

2. **Adaptive Formula**:
   ```python
   timeout = p95_latency * safety_margin

   # Where safety_margin adapts:
   safety_margin = {
       'low_load': 1.5,    # CPU < 50%
       'medium_load': 2.0,  # CPU 50-80%
       'high_load': 3.0,    # CPU > 80%
   }
   ```

3. **Thompson Sampling for Safety Margin**:
   - Each load condition has Beta distribution (α, β)
   - Sample safety margin from distribution
   - Update on timeout success/failure

4. **Success Criteria**:
   - Timeout triggered < 1% of queries (too tight if higher)
   - Average timeout headroom 30-50% (too loose if higher)

**Implementation**:
```python
class TimeoutTuner:
    """Self-tuning timeout management."""

    def __init__(self):
        self.latencies = defaultdict(deque)  # Per-stage latencies (1000 samples)
        self.bandits = {
            'low_load': ThompsonBandit(n_arms=5),    # 5 safety margins: 1.2, 1.5, 2.0, 2.5, 3.0
            'medium_load': ThompsonBandit(n_arms=5),
            'high_load': ThompsonBandit(n_arms=5),
        }
        self.load_state = 'medium_load'

    def record_stage(self, stage: str, duration_ms: float):
        """Record stage completion."""
        self.latencies[stage].append(duration_ms)
        if len(self.latencies[stage]) > 1000:
            self.latencies[stage].popleft()

    def get_adaptive_timeout(self, stage: str) -> float:
        """Get current optimal timeout for stage."""
        # Measure p95
        samples = list(self.latencies[stage])
        if len(samples) < 20:
            return DEFAULT_TIMEOUTS[stage]  # Not enough data

        p95 = np.percentile(samples, 95)

        # Select safety margin via Thompson Sampling
        bandit = self.bandits[self.load_state]
        margin_idx = bandit.sample()
        margin = SAFETY_MARGINS[margin_idx]

        timeout = p95 * margin / 1000.0  # Convert to seconds
        return timeout

    def update_from_outcome(self, stage: str, timed_out: bool, actual_duration: float):
        """Learn from timeout outcome."""
        bandit = self.bandits[self.load_state]
        margin_idx = self.last_margin_used[stage]

        if timed_out:
            # Timeout too aggressive
            bandit.update(margin_idx, success=False, confidence=0.0)
        else:
            # Success - but was headroom reasonable?
            timeout = self.last_timeout_used[stage]
            headroom = (timeout - actual_duration) / timeout

            # Ideal headroom: 30-50%
            if 0.3 <= headroom <= 0.5:
                confidence = 1.0  # Perfect
            elif headroom < 0.3:
                confidence = 0.7  # Cutting it close
            else:
                confidence = 0.5  # Too much slack (wastes time)

            bandit.update(margin_idx, success=True, confidence=confidence)
```

**Expected Impact**:
- Timeouts adapt to actual hardware (fast dev laptop vs slow CI)
- Adapt to system load (tight under low load, generous under high load)
- Reduce false timeouts from 5-10% → <1%
- Reduce wasted waiting from ~40% headroom → 30-35%

---

### Agent 2: CacheTuner

**Responsibility**: Optimize cache sizes and eviction policies
**Parameters Managed** (6):
```python
parse_cache_size: int = 10000
merge_cache_size: int = 50000
semantic_cache_size: int = 10000
working_memory_size: int = 100
episodic_buffer_size: int = 100
```

**Current Problem**:
- Fixed cache sizes regardless of available memory
- No adaptation to hit rate patterns
- Potential memory waste or cache thrashing

**Self-Tuning Strategy**:

1. **Measurement Phase**:
   - Track hit rates per cache tier (parse, merge, semantic)
   - Monitor memory pressure (available RAM)
   - Measure eviction frequency

2. **Adaptive Sizing Formula**:
   ```python
   # Target: 75% hit rate minimum
   if hit_rate < 0.75:
       cache_size *= 1.2  # Grow cache
   elif hit_rate > 0.95 and evictions < 10/hour:
       cache_size *= 0.9  # Shrink cache (wasting memory)

   # Constrain by available memory
   max_cache = available_memory_mb * 0.3  # Use max 30% RAM
   cache_size = min(cache_size, max_cache)
   ```

3. **Thompson Sampling for Size Multipliers**:
   - Arms: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0] (size multipliers)
   - Reward: hit_rate × memory_efficiency
   - Memory efficiency = hits_per_mb

**Implementation**:
```python
class CacheTuner:
    """Self-tuning cache sizing."""

    def __init__(self):
        self.hit_rates = defaultdict(deque)
        self.sizes = {
            'parse': 10000,
            'merge': 50000,
            'semantic': 10000,
        }
        self.bandits = {tier: ThompsonBandit(n_arms=6) for tier in self.sizes}

    def tune_cache_size(self, tier: str) -> int:
        """Adaptively tune cache size."""
        hit_rate = self.get_hit_rate(tier)
        memory_mb = self.estimate_memory_usage(tier)

        # Thompson Sampling for size multiplier
        bandit = self.bandits[tier]
        multiplier_idx = bandit.sample()
        multiplier = SIZE_MULTIPLIERS[multiplier_idx]

        new_size = int(self.sizes[tier] * multiplier)

        # Constrain by available memory
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        max_size = int((available_mb * 0.3) / BYTES_PER_ENTRY[tier])

        return min(new_size, max_size)

    def update_from_metrics(self, tier: str, hit_rate: float, memory_mb: float):
        """Learn from cache performance."""
        bandit = self.bandits[tier]
        multiplier_idx = self.last_multiplier_used[tier]

        # Reward: hit_rate × memory_efficiency
        hits_per_mb = (hit_rate * 1000) / memory_mb  # Normalized
        reward = hit_rate * min(1.0, hits_per_mb / 10.0)

        success = hit_rate >= 0.75  # Target minimum
        bandit.update(multiplier_idx, success=success, confidence=reward)
```

**Expected Impact**:
- Cache sizes adapt to workload (small for simple queries, large for complex)
- Memory usage optimized (no waste, no thrashing)
- Hit rates maintained at target 75-90%

---

### Agent 3: ThresholdTuner

**Responsibility**: Optimize similarity and activation thresholds
**Parameters Managed** (12):
```python
prefilter_similarity_threshold: float = 0.3
prefilter_keep_ratio: float = 0.7
spring_activation_threshold: float = 0.1
packing_activation_threshold: float = 0.3
packing_compression_threshold: float = 0.7
halt_on_low_confidence: float = 0.3
confidence_threshold: float = 0.85
recursive_learning_refinement_threshold: float = 0.75
importance_thresholds: List[float] = [0.6, 0.75, 0.85]
```

**Current Problem**:
- Fixed thresholds regardless of data distribution
- No adaptation to precision/recall tradeoffs
- Manual tuning required per domain

**Self-Tuning Strategy**:

1. **Measurement Phase**:
   - Track precision/recall at different threshold levels
   - Measure retrieval quality (relevance scores)
   - Monitor false positive/negative rates

2. **Multi-Armed Bandit per Threshold**:
   - Each threshold has 5-7 arms (candidate values)
   - Reward: F1 score or precision×recall
   - Constraints: precision ≥ 0.7, recall ≥ 0.6

3. **Adaptive Threshold Selection**:
   ```python
   # For similarity thresholds
   if precision < 0.7:
       threshold += 0.05  # More selective
   elif recall < 0.6:
       threshold -= 0.05  # More inclusive

   # Thompson Sampling for fine-tuning
   threshold = sample_from_learned_distribution(bandit)
   ```

**Implementation**:
```python
class ThresholdTuner:
    """Self-tuning threshold management."""

    def __init__(self):
        self.thresholds = {
            'similarity': 0.3,
            'activation': 0.1,
            'confidence': 0.75,
        }
        self.bandits = {
            'similarity': ThompsonBandit(n_arms=7),  # [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
            'activation': ThompsonBandit(n_arms=5),  # [0.05, 0.1, 0.15, 0.2, 0.25]
            'confidence': ThompsonBandit(n_arms=5),  # [0.7, 0.75, 0.8, 0.85, 0.9]
        }
        self.metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    def get_adaptive_threshold(self, threshold_type: str) -> float:
        """Get current optimal threshold."""
        bandit = self.bandits[threshold_type]
        arm_idx = bandit.sample()
        return THRESHOLD_CANDIDATES[threshold_type][arm_idx]

    def update_from_retrieval(self, threshold_type: str, tp: int, fp: int, fn: int):
        """Learn from retrieval precision/recall."""
        bandit = self.bandits[threshold_type]
        arm_idx = self.last_arm_used[threshold_type]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        success = precision >= 0.7 and recall >= 0.6
        bandit.update(arm_idx, success=success, confidence=f1)
```

**Expected Impact**:
- Thresholds adapt to data distribution (different per domain)
- Precision/recall balanced automatically
- No manual tuning required

---

### Agent 4: MemoryTuner

**Responsibility**: Optimize memory backend and retrieval parameters
**Parameters Managed** (11):
```python
memory_backend: MemoryBackend = MemoryBackend.HYBRID
retrieval_k: int = 6
bm25_weight: float = 0.15
hyperspace_depth: int = 3
hyperspace_breadth: int = 10
hyperspace_thresholds: List[float] = [0.6, 0.75, 0.85]
neo4j_connection_timeout: float = 30.0
```

**Current Problem**:
- Fixed retrieval_k regardless of query complexity
- BM25 weight hardcoded (should vary by query type)
- Backend selection manual

**Self-Tuning Strategy**:

1. **Backend Selection** (already partially adaptive via auto-fallback):
   - Monitor backend latencies and success rates
   - Thompson Sampling for backend selection per query type
   - Auto-switch on consistent failures

2. **Retrieval K Tuning**:
   ```python
   # Adaptive k based on query complexity
   if query_entity_count > 5:
       k = 10  # Complex query needs more context
   elif query_length < 10:
       k = 3   # Simple query needs less
   else:
       k = 6   # Default

   # Thompson Sampling for fine-tuning
   k = sample_from_learned_distribution(bandit)
   ```

3. **BM25 Weight Adaptation**:
   - Track query types (factual vs semantic)
   - Learn optimal BM25 weight per type
   - Factual queries: higher BM25 weight (keyword matching)
   - Semantic queries: lower BM25 weight (embedding similarity)

**Implementation**:
```python
class MemoryTuner:
    """Self-tuning memory system parameters."""

    def __init__(self):
        self.retrieval_k_bandit = ThompsonBandit(n_arms=5)  # k ∈ {3, 6, 10, 15, 20}
        self.bm25_weight_bandit = ThompsonBandit(n_arms=5)  # weight ∈ {0.05, 0.1, 0.15, 0.2, 0.25}
        self.backend_bandit = ThompsonBandit(n_arms=3)      # INMEMORY, HYBRID, HYPERSPACE

    def get_adaptive_k(self, query_complexity: str) -> int:
        """Adaptive retrieval k."""
        arm = self.retrieval_k_bandit.sample()
        return K_CANDIDATES[arm]

    def update_from_retrieval(self, k_used: int, relevance_scores: List[float]):
        """Learn from retrieval quality."""
        # Measure retrieval quality
        avg_relevance = np.mean(relevance_scores)
        waste = sum(1 for r in relevance_scores if r < 0.3) / len(relevance_scores)

        # Reward: high relevance, low waste
        reward = avg_relevance * (1.0 - waste)

        arm_idx = K_CANDIDATES.index(k_used)
        success = avg_relevance >= 0.6 and waste < 0.3
        self.retrieval_k_bandit.update(arm_idx, success=success, confidence=reward)
```

**Expected Impact**:
- Retrieval k adapts to query complexity
- BM25 weight optimized per query type
- Backend selection learned from performance

---

### Agent 5: ComplexityTuner

**Responsibility**: Auto-select execution mode (BARE/FAST/FUSED)
**Parameters Managed** (8):
```python
mode: ExecutionMode = ExecutionMode.FUSED
n_transformer_layers: int = 2
n_attention_heads: int = 4
scales: List[int] = [768]
enable_linguistic_gate: bool = False
enable_semantic_calculus: bool = False
```

**Current Problem**:
- Manual mode selection
- Fixed complexity regardless of query difficulty
- No cost/benefit analysis

**Self-Tuning Strategy**:

1. **Query Complexity Scoring**:
   ```python
   complexity_score = (
       0.3 * query_length_norm +
       0.3 * entity_count_norm +
       0.2 * question_depth_norm +
       0.2 * semantic_richness_norm
   )

   if complexity_score < 0.3:
       mode = BARE   # Simple factual query
   elif complexity_score < 0.7:
       mode = FAST   # Standard query
   else:
       mode = FUSED  # Complex analytical query
   ```

2. **Thompson Sampling for Boundary Tuning**:
   - Learn optimal boundary values (0.3, 0.7)
   - Track mode selection success rate
   - Reward: confidence / latency tradeoff

3. **Cost-Benefit Analysis**:
   ```python
   benefit = confidence_gain
   cost = latency_increase_ms / 1000.0

   utility = benefit - (cost * cost_sensitivity)

   if utility > 0:
       mode = FUSED  # Worth the cost
   else:
       mode = FAST   # Not worth it
   ```

**Implementation**:
```python
class ComplexityTuner:
    """Self-tuning execution mode selection."""

    def __init__(self):
        self.boundary_bandit = ThompsonBandit(n_arms=9)  # 9 boundary pairs
        self.cost_sensitivity = 1.0  # Adapt based on user preference

    def select_mode(self, query: Query) -> ExecutionMode:
        """Adaptively select execution mode."""
        complexity = self.compute_complexity(query)

        # Sample boundaries from learned distribution
        arm_idx = self.boundary_bandit.sample()
        low_bound, high_bound = BOUNDARY_CANDIDATES[arm_idx]

        if complexity < low_bound:
            return ExecutionMode.BARE
        elif complexity < high_bound:
            return ExecutionMode.FAST
        else:
            return ExecutionMode.FUSED

    def update_from_outcome(self, mode: ExecutionMode, confidence: float, latency_ms: float):
        """Learn from mode selection outcome."""
        arm_idx = self.last_arm_used

        # Cost-benefit analysis
        benefit = confidence
        cost = latency_ms / 1000.0
        utility = benefit - (cost * self.cost_sensitivity)

        success = utility > 0 and confidence >= 0.75
        self.boundary_bandit.update(arm_idx, success=success, confidence=abs(utility))
```

**Expected Impact**:
- Mode selection adapts to query characteristics
- Cost/benefit optimized per user preference
- Latency reduced 20-40% by avoiding over-complexity

---

### Agent 6: PolicyTuner

**Responsibility**: Optimize policy and exploration parameters
**Parameters Managed** (4):
```python
epsilon: float = 0.1
blend_neural_weight: float = 0.7
bandit_strategy: BanditStrategy = BanditStrategy.EPSILON_GREEDY
learning_rate: float = 3e-4
```

**Current Problem**:
- Fixed exploration rate (10%)
- No adaptation to exploration/exploitation tradeoff
- Learning rate not tuned to convergence speed

**Self-Tuning Strategy**:

1. **Adaptive Epsilon**:
   ```python
   # Decay epsilon as system gains confidence
   epsilon = epsilon_initial * exp(-decay_rate * total_queries)

   # Minimum epsilon for perpetual exploration
   epsilon = max(epsilon, epsilon_min)  # e.g., 0.02
   ```

2. **Thompson Sampling for Blend Weight**:
   - Track neural vs bandit performance
   - Learn optimal blend weight per query type
   - Factual queries: higher neural weight
   - Uncertain domains: higher bandit weight

3. **Learning Rate Adaptation**:
   ```python
   # Adaptive learning rate based on convergence
   if policy_loss_variance < threshold:
       learning_rate *= 0.95  # Converging, reduce LR
   elif policy_loss_variance > threshold * 2:
       learning_rate *= 1.05  # Diverging, increase LR
   ```

**Implementation**:
```python
class PolicyTuner:
    """Self-tuning policy parameters."""

    def __init__(self):
        self.epsilon = 0.1
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.02
        self.blend_bandit = ThompsonBandit(n_arms=5)  # [0.5, 0.6, 0.7, 0.8, 0.9]
        self.total_queries = 0

    def get_adaptive_epsilon(self) -> float:
        """Adaptive exploration rate."""
        self.total_queries += 1
        epsilon = self.epsilon * (self.epsilon_decay ** self.total_queries)
        return max(epsilon, self.epsilon_min)

    def get_adaptive_blend_weight(self, query_type: str) -> float:
        """Adaptive neural/bandit blend."""
        arm_idx = self.blend_bandit.sample()
        return BLEND_WEIGHTS[arm_idx]

    def update_from_outcome(self, blend_weight: float, confidence: float, tool_correct: bool):
        """Learn from policy decision."""
        arm_idx = BLEND_WEIGHTS.index(blend_weight)

        success = tool_correct and confidence >= 0.75
        self.blend_bandit.update(arm_idx, success=success, confidence=confidence)
```

**Expected Impact**:
- Exploration adapts to system maturity
- Blend weight optimized per query type
- Faster convergence to optimal policies

---

### Agent 7: PhysicsTuner

**Responsibility**: Optimize spring dynamics and semantic parameters
**Parameters Managed** (7):
```python
spring_stiffness: float = 0.15
spring_damping: float = 0.85
spring_decay: float = 0.98
spring_iterations: int = 200
spring_convergence_epsilon: float = 1e-4
semantic_dt: float = 1.0
```

**Current Problem**:
- All hardcoded physics constants
- No adaptation to graph topology
- Potential over/under-damping

**Self-Tuning Strategy**:

1. **Convergence Speed Analysis**:
   ```python
   # Measure iterations to convergence
   actual_iterations = count_until_convergence()

   if actual_iterations < 50:
       # Over-damped, increase stiffness
       stiffness *= 1.1
   elif actual_iterations > 150:
       # Under-damped, increase damping
       damping *= 1.05
   ```

2. **Thompson Sampling for Parameter Combinations**:
   - Sample (stiffness, damping) pairs
   - Reward: convergence_speed × activation_quality
   - Quality = relevance of activated nodes

3. **Adaptive Iteration Limit**:
   ```python
   # Set limit to p95 convergence time + 20%
   max_iterations = int(p95_iterations * 1.2)
   ```

**Implementation**:
```python
class PhysicsTuner:
    """Self-tuning physics parameters."""

    def __init__(self):
        self.param_bandit = ThompsonBandit(n_arms=25)  # 5×5 grid of (stiffness, damping)
        self.convergence_times = deque(maxlen=100)

    def get_adaptive_params(self) -> Tuple[float, float]:
        """Get optimal stiffness and damping."""
        arm_idx = self.param_bandit.sample()
        stiffness_idx, damping_idx = divmod(arm_idx, 5)

        stiffness = STIFFNESS_CANDIDATES[stiffness_idx]
        damping = DAMPING_CANDIDATES[damping_idx]

        return stiffness, damping

    def update_from_convergence(self, stiffness: float, damping: float,
                                iterations: int, activation_quality: float):
        """Learn from spring dynamics outcome."""
        arm_idx = self.get_arm_index(stiffness, damping)

        # Reward: fast convergence + high quality
        speed_score = 1.0 - (iterations / 200.0)  # Normalized
        reward = 0.6 * speed_score + 0.4 * activation_quality

        success = iterations < 150 and activation_quality >= 0.7
        self.param_bandit.update(arm_idx, success=success, confidence=reward)
```

**Expected Impact**:
- Physics parameters adapt to graph topology
- Faster convergence (150 → 80 iterations avg)
- Higher quality activation patterns

---

## III. Master Tuning Coordinator

### Thompson Sampling for Agent Activation

**Problem**: Which tuning agent should run next?

**Solution**: Meta-bandit that selects tuning agents based on impact

```python
class MasterTuningCoordinator:
    """Coordinates all tuning agents using Thompson Sampling."""

    def __init__(self):
        self.agents = {
            'timeout': TimeoutTuner(),
            'cache': CacheTuner(),
            'threshold': ThresholdTuner(),
            'memory': MemoryTuner(),
            'complexity': ComplexityTuner(),
            'policy': PolicyTuner(),
            'physics': PhysicsTuner(),
        }
        self.meta_bandit = ThompsonBandit(n_arms=7)  # One per agent
        self.agent_impact = defaultdict(list)

    async def run_tuning_cycle(self):
        """Run one tuning iteration."""
        # Sample which agent to activate
        agent_idx = self.meta_bandit.sample()
        agent_name = list(self.agents.keys())[agent_idx]
        agent = self.agents[agent_name]

        # Measure baseline performance
        baseline_metrics = await self.measure_system_performance()

        # Run agent tuning
        tuning_result = await agent.tune()

        # Measure improvement
        new_metrics = await self.measure_system_performance()

        # Calculate impact
        impact = self.calculate_impact(baseline_metrics, new_metrics)

        # Update meta-bandit
        success = impact > 0.0
        self.meta_bandit.update(agent_idx, success=success, confidence=abs(impact))

        # Log tuning decision
        logger.info(f"Tuned {agent_name}: impact={impact:.3f}")

    def calculate_impact(self, baseline, new) -> float:
        """Calculate tuning impact score."""
        # Weighted combination of improvements
        latency_improvement = (baseline['latency'] - new['latency']) / baseline['latency']
        confidence_improvement = (new['confidence'] - baseline['confidence']) / baseline['confidence']
        cache_hit_improvement = (new['cache_hit_rate'] - baseline['cache_hit_rate']) / baseline['cache_hit_rate']

        impact = (
            0.4 * latency_improvement +
            0.4 * confidence_improvement +
            0.2 * cache_hit_improvement
        )

        return impact
```

**Impact Prioritization**:
- Agents with high recent impact get selected more often
- Low-impact agents deprioritized
- Ensures tuning effort focused on bottlenecks

---

## IV. Safe Tuning Guarantees

### Never Break the System

**Safety Mechanisms**:

1. **Bounded Parameter Ranges**:
   ```python
   # Every tuned parameter has safe min/max
   SAFE_RANGES = {
       'timeout': (0.05, 10.0),      # 50ms - 10s
       'cache_size': (100, 1000000),  # 100 - 1M entries
       'threshold': (0.0, 1.0),       # 0-1 normalized
       'epsilon': (0.01, 0.5),        # 1-50% exploration
   }

   def apply_tuning(param, new_value):
       min_val, max_val = SAFE_RANGES[param]
       safe_value = np.clip(new_value, min_val, max_val)
       return safe_value
   ```

2. **Gradual Changes Only**:
   ```python
   # Maximum change per tuning cycle
   MAX_CHANGE_PERCENT = 0.2  # 20% max change

   new_value = current_value * (1 + change_percent)
   change_percent = np.clip(change_percent, -0.2, 0.2)
   ```

3. **Rollback on Degradation**:
   ```python
   if new_metrics['confidence'] < baseline_metrics['confidence'] * 0.9:
       # Rollback tuning if confidence drops >10%
       rollback_parameter(param, previous_value)
       logger.warning(f"Rolled back {param} due to degradation")
   ```

4. **Testing Mode First**:
   ```python
   # Test tuning on 10% of queries before full deployment
   if random.random() < 0.1:
       use_tuned_parameter()
   else:
       use_default_parameter()

   # Deploy only if testing succeeds
   if test_success_rate >= 0.95:
       deploy_tuned_parameter()
   ```

5. **Circuit Breaker**:
   ```python
   class TuningCircuitBreaker:
       """Halt tuning if system degraded."""

       def __init__(self):
           self.failure_count = 0
           self.failure_threshold = 3
           self.open = False

       def check(self, metrics):
           if metrics['error_rate'] > 0.05:  # 5% errors
               self.failure_count += 1
               if self.failure_count >= self.failure_threshold:
                   self.open = True
                   logger.error("Circuit breaker OPEN - halting tuning")
           else:
               self.failure_count = 0

       def allow_tuning(self) -> bool:
           return not self.open
   ```

---

## V. Persistence & Multi-Session Learning

### Learned Parameters Survive Restarts

**Storage Format** (JSON):
```json
{
  "tuning_version": "1.0.0",
  "last_updated": "2025-11-02T12:34:56Z",
  "total_queries_seen": 15234,
  "learned_parameters": {
    "timeout_tuner": {
      "pipeline_timeout": 4.2,
      "retrieval_timeout": 0.15,
      "stage_timeouts": {
        "features": 1.8,
        "retrieval": 1.5,
        "decision": 0.6,
        "execution": 0.3
      },
      "bandit_priors": {
        "low_load": {"alpha": [12.5, 8.3, 5.1, 3.2, 2.1], "beta": [3.2, 4.1, 5.5, 8.9, 12.0]},
        "medium_load": {"alpha": [8.1, 15.2, 9.3, 4.2, 1.8], "beta": [5.2, 3.1, 4.5, 9.2, 15.3]},
        "high_load": {"alpha": [2.1, 4.5, 8.9, 18.3, 11.2], "beta": [18.2, 12.5, 7.3, 3.5, 2.1]}
      }
    },
    "cache_tuner": {
      "parse_cache_size": 12500,
      "merge_cache_size": 58000,
      "semantic_cache_size": 9500
    },
    "threshold_tuner": {
      "prefilter_similarity_threshold": 0.32,
      "activation_threshold": 0.12,
      "confidence_threshold": 0.78
    }
  },
  "meta_bandit": {
    "alpha": [15.2, 18.5, 12.3, 9.8, 14.1, 8.2, 5.3],
    "beta": [8.3, 5.2, 9.1, 11.5, 7.8, 12.9, 15.2]
  }
}
```

**Load on Startup**:
```python
def load_learned_parameters():
    """Restore learned parameters from disk."""
    path = Path("./tuning_state/learned_params.json")

    if not path.exists():
        logger.info("No learned parameters found, using defaults")
        return DEFAULT_CONFIG

    with open(path) as f:
        learned = json.load(f)

    logger.info(f"Loaded learned parameters ({learned['total_queries_seen']} queries)")
    return learned
```

**Save Periodically**:
```python
async def save_learned_parameters():
    """Persist learned parameters to disk."""
    state = {
        'tuning_version': '1.0.0',
        'last_updated': datetime.now().isoformat(),
        'total_queries_seen': coordinator.total_queries,
        'learned_parameters': {
            agent_name: agent.get_state()
            for agent_name, agent in coordinator.agents.items()
        },
        'meta_bandit': coordinator.meta_bandit.get_state(),
    }

    path = Path("./tuning_state/learned_params.json")
    path.parent.mkdir(exist_ok=True)

    with open(path, 'w') as f:
        json.dump(state, f, indent=2)

    logger.info("Saved learned parameters to disk")
```

---

## VI. Observability & Debugging

### Prometheus Metrics for Tuning Decisions

**New Metrics**:
```python
# Tuning activity
tuning_cycles_total = Counter(
    'hololoom_tuning_cycles_total',
    'Total tuning cycles executed',
    ['agent_name', 'outcome']  # outcome: success, failure, rollback
)

tuning_parameter_value = Gauge(
    'hololoom_tuning_parameter_value',
    'Current tuned parameter value',
    ['agent_name', 'parameter_name']
)

tuning_impact = Histogram(
    'hololoom_tuning_impact',
    'Impact of tuning cycle',
    ['agent_name'],
    buckets=[-0.5, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0]
)

tuning_bandit_arms = Gauge(
    'hololoom_tuning_bandit_arms',
    'Thompson Sampling arm statistics',
    ['agent_name', 'arm_index', 'stat']  # stat: alpha, beta, expected_reward
)
```

**Dashboard Panels**:
1. **Tuning Activity Timeline**: Which agents ran when
2. **Parameter Evolution**: How parameters changed over time
3. **Impact Scores**: Which tuning decisions had biggest impact
4. **Bandit Arm Distribution**: Which arms getting explored/exploited
5. **Rollback Events**: When tuning was reverted

**Logging**:
```python
logger.info("Tuning cycle started", extra={
    'agent': 'timeout_tuner',
    'baseline_latency': 145.2,
    'baseline_confidence': 0.82,
})

logger.info("Tuning cycle completed", extra={
    'agent': 'timeout_tuner',
    'new_latency': 138.5,
    'new_confidence': 0.84,
    'impact': 0.15,
    'parameters_changed': {'retrieval_timeout': (0.2, 0.15)},
})
```

---

## VII. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Goal**: Infrastructure for tuning agents

- [ ] Create `TuningAgent` base class
- [ ] Implement `MasterTuningCoordinator`
- [ ] Add persistence layer (JSON storage)
- [ ] Create Prometheus metrics
- [ ] Set up safe parameter ranges

**Deliverable**: Framework ready, no agents yet

---

### Phase 2: Agent 1-3 (Week 3-4)

**Goal**: Deploy first 3 tuning agents

- [ ] Implement `TimeoutTuner`
- [ ] Implement `CacheTuner`
- [ ] Implement `ThresholdTuner`
- [ ] Test on 1000 queries
- [ ] Measure baseline vs tuned performance

**Success Criteria**:
- Timeouts adapt to hardware
- Cache hit rates improve 5-10%
- Thresholds optimize precision/recall

---

### Phase 3: Agent 4-5 (Week 5-6)

**Goal**: Memory and complexity tuning

- [ ] Implement `MemoryTuner`
- [ ] Implement `ComplexityTuner`
- [ ] Integrate with existing complexity detection
- [ ] Test cost/benefit tradeoffs

**Success Criteria**:
- Retrieval k adapts to query complexity
- Mode selection reduces latency 20-30%

---

### Phase 4: Agent 6-7 (Week 7-8)

**Goal**: Policy and physics tuning

- [ ] Implement `PolicyTuner`
- [ ] Implement `PhysicsTuner`
- [ ] Full system integration test
- [ ] Production deployment

**Success Criteria**:
- Epsilon adapts to system maturity
- Spring dynamics converge faster

---

### Phase 5: Production Monitoring (Week 9-10)

**Goal**: Observability and optimization

- [ ] Create tuning dashboard (Grafana)
- [ ] Set up alerting on degradation
- [ ] Tune meta-bandit impact weighting
- [ ] Document tuning behavior

**Success Criteria**:
- Full observability of tuning decisions
- Zero production incidents from tuning

---

## VIII. Expected Outcomes

### Quantitative Goals

| Metric | Baseline | After Tuning | Improvement |
|--------|----------|--------------|-------------|
| **Configuration Parameters** | 72 | 3 | -96% |
| **Manual Tuning Required** | Always | Never | -100% |
| **Timeout False Positives** | 5-10% | <1% | -80% |
| **Cache Hit Rate** | 60-70% | 75-90% | +20% |
| **Mode Selection Latency** | Fixed | Adaptive | -25% avg |
| **Precision/Recall** | Fixed | Optimized | +10-15% F1 |
| **Parameter Waste** | 26% | 0% | -100% |

### Qualitative Goals

1. **Zero-Configuration Experience**:
   - User sets 3 knobs: mode, environment, backend
   - System learns rest from production data

2. **Hardware-Agnostic**:
   - Same config works on fast laptop and slow CI
   - Timeouts adapt to actual performance

3. **Domain-Adaptive**:
   - Thresholds learn from data distribution
   - No manual tuning per domain

4. **Self-Healing**:
   - Degradation detected automatically
   - Rollback on failures
   - Circuit breaker prevents cascading failures

5. **Transparent**:
   - All tuning decisions logged
   - Prometheus metrics expose internal state
   - Grafana dashboard for visualization

---

## IX. The Elegant End State

### Configuration Level 1: Zero-Configuration

**User Experience**:
```python
from HoloLoom import HoloLoom

# That's it. Everything else is learned.
async with HoloLoom() as loom:
    result = await loom.recall("What is Thompson Sampling?")
```

**Behind the Scenes**:
- Master coordinator running background tuning cycles
- 7 tuning agents continuously optimizing
- Thompson Sampling exploring parameter space
- Learned parameters persisted across sessions
- Prometheus metrics tracking all decisions
- Circuit breaker preventing failures

**The Result**:
- Timeouts adapt to hardware: 150ms on laptop, 450ms on CI
- Cache sizes adapt to memory: 50K on laptop, 10K on Pi
- Thresholds adapt to domain: 0.3 for factual, 0.7 for creative
- Mode selection adapts to query: BARE for simple, FUSED for complex
- Exploration adapts to maturity: 10% early, 2% after 10K queries

**The Philosophy**:
> *"The system configures itself based on what actually works, not on what we think should work."*

---

## X. Beyond Configuration: Self-Improving Intelligence

**This is more than parameter tuning.**

This is the foundation for **recursive self-improvement**:

1. **Agents tune parameters** → Better performance
2. **Better performance** → Higher quality data
3. **Higher quality data** → Better agent learning
4. **Better agent learning** → Smarter tuning decisions
5. **Loop continues indefinitely**

**The moonshot**: A system that gets smarter with every query, requires zero manual tuning, and adapts to any environment.

**Status**: Ready to deploy agent swarm.

---

**End of Strategy Document**

**Next Steps**:
1. Review and approve strategy
2. Deploy Phase 1 (foundation) this week
3. Begin Agent 1 implementation (TimeoutTuner)
4. Measure baseline metrics for comparison

🚀 **Moonshot Level 3: Self-Tuning Intelligence**
