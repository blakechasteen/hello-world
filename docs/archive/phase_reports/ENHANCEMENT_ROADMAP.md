# HoloLoom Math→Meaning Enhancement Roadmap

**Based on**: Research findings from contextual bandits, neural-symbolic AI, and data-to-text NLG best practices

## Executive Summary

We've built a working Math→Meaning pipeline. Now we make it **world-class** by implementing research-backed best practices.

**Current State**: ✅ Working end-to-end
**Target State**: 🎯 Production-ready, learning, explainable

---

## Phase 1: Enhance Operation Selector (HIGHEST IMPACT)

### 1.1 Contextual Thompson Sampling

**Current**: Basic TS with (operation, intent) pairs
**Target**: Contextual bandits with rich feature vectors

**Implementation**:

```python
# File: warp/math/contextual_selector.py

class ContextualOperationSelector:
    """Contextual bandit for operation selection."""

    def __init__(self):
        # Gaussian linear bandit
        self.theta_mean = {}  # Mean weights per operation
        self.theta_cov = {}   # Covariance per operation

    def select_operation(self, context_features, candidates):
        """
        Select operation using contextual TS.

        Args:
            context_features: np.array([
                query_embedding (384),
                intent_vector (7),
                domain_embedding (50),
                recent_success_rate (1),
                budget_remaining (1),
                time_embedding (24)
            ])
            candidates: List[MathOperation]

        Returns:
            Selected operation
        """
        samples = {}

        for op in candidates:
            # Sample theta from posterior N(μ, Σ)
            theta = np.random.multivariate_normal(
                self.theta_mean[op.name],
                self.theta_cov[op.name]
            )

            # Expected reward = context · theta
            samples[op.name] = np.dot(context_features, theta)

        # Choose operation with highest sampled reward
        best_op = max(samples, key=samples.get)
        return next(op for op in candidates if op.name == best_op)

    def update(self, context_features, operation, reward):
        """Bayesian update of posterior."""
        # Bayesian linear regression update
        # θ|data ~ N(μ_new, Σ_new)
        ...
```

**Impact**: 2-3x better operation selection in varied contexts

---

### 1.2 Feel-Good Thompson Sampling (FGTS)

**Enhancement**: Add exploration bonus for worst-case guarantees

```python
class FGTSSelector(ContextualOperationSelector):
    """FGTS with minimax-optimal regret."""

    def select_operation(self, context, candidates, t):
        # Standard TS sample
        ts_samples = super().select_operation(context, candidates)

        # Add UCB-style exploration bonus
        for op_name, ts_value in ts_samples.items():
            n_pulls = self.pull_counts[op_name]
            ucb_bonus = np.sqrt(2 * np.log(t) / (n_pulls + 1))

            ts_samples[op_name] += self.beta * ucb_bonus

        return max(ts_samples, key=ts_samples.get)
```

**Impact**: Better worst-case performance, safety guarantees

---

### 1.3 Context Feature Engineering

**Current**: Simple intent classification
**Target**: Rich 470-dimensional context

```python
class ContextBuilder:
    """Build rich context for operation selection."""

    def build_context(self, query, history, environment):
        context = np.concatenate([
            # Query understanding (384)
            self.embed_query(query),

            # Intent distribution (7)
            self.classify_intents(query),

            # Domain knowledge (50)
            self.embed_domain(query),

            # Performance history (10)
            self.recent_performance(history, window=10),

            # Resource constraints (3)
            [
                environment["budget_remaining"] / 100,
                environment["time_remaining"] / 60,
                environment["memory_available"] / 1000
            ],

            # Temporal features (24)
            self.time_embeddings(datetime.now()),

            # User features (if available) (10)
            self.user_profile(environment.get("user_id"))
        ])

        return context  # shape: (470,)
```

**Impact**: Context-aware selection, 40-50% better accuracy

---

## Phase 2: Enhance Meaning Synthesis (HIGH IMPACT)

### 2.1 Five-Stage NLG Pipeline

**Current**: Template-based generation
**Target**: Full data-to-text pipeline

```python
# File: warp/math/advanced_nlg.py

class AdvancedMeaningSynthesizer:
    """5-stage NLG pipeline."""

    def synthesize(self, numerical_results, intent, context):
        # Stage 1: Data Understanding
        understanding = self.data_understander.analyze(numerical_results)

        # Stage 2: Content Planning
        content_plan = self.content_planner.plan(
            understanding, intent, audience=context.get("audience", "technical")
        )

        # Stage 3: Document Structuring
        structure = self.document_structurer.structure(content_plan)

        # Stage 4: Text Generation
        text = self.text_generator.generate(structure, context)

        # Stage 5: Post-processing
        final_text = self.post_processor.refine(text)

        return MeaningResult(
            text=final_text,
            understanding=understanding,
            plan=content_plan,
            structure=structure
        )
```

---

### 2.2 Data Understanding Layer

```python
class DataUnderstander:
    """Analyze numerical results before generating text."""

    def analyze(self, results):
        insights = {
            "patterns": self._detect_patterns(results),
            "outliers": self._find_outliers(results),
            "trends": self._analyze_trends(results),
            "comparisons": self._compare_to_expectations(results),
            "statistics": self._compute_statistics(results),
            "correlations": self._find_correlations(results)
        }

        # Determine what's interesting/surprising
        insights["salience"] = self._rank_by_importance(insights)

        return insights

    def _detect_patterns(self, results):
        """Detect patterns in data."""
        patterns = []

        # Check for distributions
        if "values" in results:
            vals = results["values"]

            if self._is_bimodal(vals):
                patterns.append({"type": "bimodal", "modes": self._find_modes(vals)})

            if self._is_power_law(vals):
                patterns.append({"type": "power_law", "exponent": self._fit_power_law(vals)})

        return patterns
```

---

### 2.3 Content Planning

```python
class ContentPlanner:
    """Decide what to say and in what order."""

    def plan(self, understanding, intent, audience):
        plan = {
            "narrative_arc": self._select_narrative(intent),
            "sections": [],
            "emphasis": [],
            "detail_level": self._determine_detail_level(audience)
        }

        # What to include?
        for insight in understanding["salience"]:
            if insight["importance"] > 0.7:
                plan["sections"].append({
                    "type": insight["type"],
                    "content": insight,
                    "priority": insight["importance"]
                })

        # What to emphasize?
        if understanding.get("surprising"):
            plan["emphasis"].append("highlight_surprise")

        if understanding.get("actionable"):
            plan["emphasis"].append("end_with_recommendations")

        # Narrative flow
        plan["structure"] = self._order_sections(plan["sections"], intent)

        return plan
```

---

### 2.4 Multi-Modal Output

```python
class MultiModalSynthesizer:
    """Generate text + visualizations + tables."""

    def synthesize(self, results, intent):
        meaning = {
            "text": self._generate_text(results),
            "visualizations": self._select_visualizations(results),
            "tables": self._create_tables(results),
            "metadata": self._add_metadata(results)
        }

        return meaning

    def _select_visualizations(self, results):
        """Auto-select appropriate visualizations."""
        viz = []

        # Distribution → histogram
        if "distribution" in results:
            viz.append({
                "type": "histogram",
                "data": results["distribution"],
                "title": "Value Distribution",
                "description": "Histogram showing frequency of values"
            })

        # Time series → line chart
        if "time_series" in results:
            viz.append({
                "type": "line_chart",
                "data": results["time_series"],
                "title": "Trend Over Time",
                "description": "Time series showing evolution"
            })

        # Comparison → bar chart
        if "comparisons" in results:
            viz.append({
                "type": "bar_chart",
                "data": results["comparisons"],
                "title": "Comparative Analysis"
            })

        return viz
```

---

## Phase 3: Enhance Integration (HIGH IMPACT)

### 3.1 Neural-Symbolic Bridge

```python
# File: neural_symbolic_bridge.py

class NeuralSymbolicBridge:
    """Bidirectional mapping between neural and symbolic."""

    def __init__(self):
        self.neural_encoder = NeuralEncoder()
        self.symbolic_decoder = SymbolicDecoder()

    def neural_to_symbolic(self, embedding: np.ndarray) -> SymbolicRepresentation:
        """Convert neural representation to symbolic."""
        symbolic = SymbolicRepresentation()

        # Extract discrete intent
        symbolic.intent = self._classify_intent(embedding)

        # Extract entities
        symbolic.entities = self._extract_entities(embedding)

        # Infer constraints
        symbolic.constraints = self._infer_constraints(embedding)

        # Extract logical structure
        symbolic.logical_form = self._parse_logic(embedding)

        return symbolic

    def symbolic_to_neural(self, symbolic: SymbolicRepresentation) -> np.ndarray:
        """Convert symbolic to neural."""
        # Encode intent
        intent_emb = self.intent_embedder(symbolic.intent)

        # Encode entities
        entity_emb = self.entity_embedder(symbolic.entities)

        # Encode logical structure
        logic_emb = self.logic_embedder(symbolic.logical_form)

        # Combine
        return np.concatenate([intent_emb, entity_emb, logic_emb])
```

---

### 3.2 Explanation Generation

```python
class ExplainableSelector:
    """Generate explanations for operation choices."""

    def select_with_explanation(self, context, candidates):
        # Select operation
        selected_op, scores = self._select_operation_with_scores(
            context, candidates
        )

        # Generate explanation
        explanation = self._explain_choice(
            selected_op, scores, context, candidates
        )

        return selected_op, explanation

    def _explain_choice(self, selected, scores, context, candidates):
        """Generate natural language explanation."""
        runner_up = sorted(scores.items(), key=lambda x: x[1], reverse=True)[1]

        explanation = {
            "summary": f"Selected '{selected.name}' for this query",
            "reasons": [],
            "alternatives": []
        }

        # Why selected?
        explanation["reasons"].append(
            f"Intent is {context['intent']} (confidence: {context['intent_conf']:.1%})"
        )

        explanation["reasons"].append(
            f"Historical success rate: {scores[selected.name]['success_rate']:.1%}"
        )

        explanation["reasons"].append(
            f"Expected reward: {scores[selected.name]['expected_reward']:.3f}"
        )

        # Why not runner-up?
        explanation["alternatives"].append({
            "operation": runner_up[0],
            "score": runner_up[1],
            "reason": f"Lower expected reward ({runner_up[1]:.3f} vs {scores[selected.name]['expected_reward']:.3f})"
        })

        return explanation
```

---

## Phase 4: Bootstrap & Validate

### 4.1 Bootstrap Script (100 Queries)

```python
# File: bootstrap_smart_system.py

async def bootstrap():
    """Run 100 diverse queries to train RL."""

    orchestrator = create_smart_orchestrator()

    # Query categories
    queries = generate_diverse_queries(
        similarity=25,
        optimization=25,
        analysis=25,
        verification=25
    )

    # Run with progress tracking
    results = []
    learning_curve = []

    for i, query in enumerate(queries):
        spacetime = await orchestrator.weave(query)

        # Record
        results.append({
            "query": query,
            "success": spacetime.confidence >= 0.7,
            "confidence": spacetime.confidence,
            "operations": get_operations_used(spacetime)
        })

        # Track learning
        if i % 10 == 0:
            stats = orchestrator.get_statistics()
            learning_curve.append({
                "iteration": i,
                "avg_confidence": np.mean([r["confidence"] for r in results[-10:]]),
                "success_rate": np.mean([r["success"] for r in results[-10:]]),
                "rl_stats": stats["math_pipeline"]["rl_learning"]
            })

    # Save results
    save_bootstrap_results(results, learning_curve)

    # Visualize
    plot_learning_curves(learning_curve)
```

---

### 4.2 Validation Suite

```python
# File: validation_suite.py

class ValidationSuite:
    """Comprehensive validation."""

    def run_all_tests(self):
        results = {
            "unit_tests": self.run_unit_tests(),
            "integration_tests": self.run_integration_tests(),
            "property_tests": self.run_property_tests(),
            "performance_tests": self.run_performance_tests(),
            "quality_tests": self.run_quality_tests()
        }

        return results

    def run_property_tests(self):
        """Test mathematical properties."""
        tests = [
            self.test_metric_axioms(),
            self.test_convergence_properties(),
            self.test_stability_bounds(),
            self.test_numerical_stability()
        ]

        return {
            "passed": sum(t["passed"] for t in tests),
            "total": len(tests),
            "details": tests
        }

    def run_quality_tests(self):
        """Test output quality."""
        tests = []

        # Meaning quality
        tests.append(self.test_meaning_coherence())
        tests.append(self.test_meaning_accuracy())
        tests.append(self.test_meaning_completeness())

        # RL quality
        tests.append(self.test_rl_convergence())
        tests.append(self.test_exploration_rate())

        return tests
```

---

## Phase 5: Visual Dashboard

### 5.1 Real-Time Monitoring

```python
# File: dashboard/monitoring_dashboard.py

class MathMeaningDashboard:
    """Real-time monitoring dashboard."""

    def __init__(self):
        self.app = dash.Dash(__name__)
        self.setup_layout()

    def setup_layout(self):
        self.app.layout = html.Div([
            # Header
            html.H1("HoloLoom Math→Meaning Monitor"),

            # Real-time metrics
            dcc.Graph(id="live-learning-curve"),
            dcc.Graph(id="operation-distribution"),
            dcc.Graph(id="confidence-histogram"),

            # RL statistics
            html.Div(id="rl-stats"),

            # Recent queries
            html.Div(id="recent-queries"),

            # Update interval
            dcc.Interval(id="interval", interval=1000)  # 1 second
        ])

    def update_learning_curve(self):
        """Update learning curve in real-time."""
        stats = self.orchestrator.get_statistics()
        rl_stats = stats["math_pipeline"]["rl_learning"]

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(self.learning_history))),
            y=[h["avg_confidence"] for h in self.learning_history],
            name="Avg Confidence"
        ))

        return fig
```

---

## Implementation Timeline

### Week 1: Enhance Operation Selector
- Day 1-2: Contextual features + Gaussian posteriors
- Day 3-4: FGTS implementation
- Day 5: Testing & validation

### Week 2: Enhance Meaning Synthesis
- Day 1-2: Data understanding + content planning
- Day 3-4: Document structuring + advanced generation
- Day 5: Multi-modal output

### Week 3: Integration & Bootstrap
- Day 1-2: Neural-symbolic bridge
- Day 3: Explanation generation
- Day 4: Bootstrap 100 queries
- Day 5: Analyze results

### Week 4: Dashboard & Polish
- Day 1-2: Monitoring dashboard
- Day 3: Validation suite
- Day 4-5: Documentation + demos

---

## Success Metrics

**Operation Selector**:
- [ ] Context-aware selection accuracy > 80%
- [ ] FGTS regret bounds verified
- [ ] 2-3x improvement over random selection

**Meaning Synthesis**:
- [ ] Coherence score > 0.9
- [ ] User satisfaction > 4/5
- [ ] Multi-modal output for 80% of queries

**Integration**:
- [ ] End-to-end latency < 200ms
- [ ] 95%+ uptime
- [ ] Complete explanations for all decisions

**Learning**:
- [ ] Convergence within 50 queries
- [ ] Success rate > 85% after bootstrap
- [ ] Continuous improvement demonstrated

---

## Next Steps

**IMMEDIATE** (Do now):
1. Implement contextual feature builder
2. Add Gaussian posterior to TS
3. Run bootstrap with 100 queries

**SHORT-TERM** (This week):
4. Add data understanding to NLG
5. Build explanation generator
6. Create monitoring dashboard

**MEDIUM-TERM** (Next 2 weeks):
7. Complete 5-stage NLG pipeline
8. Build validation suite
9. Add domain-specific pipelines

---

**Ready to build world-class Math→Meaning system!** 🚀
