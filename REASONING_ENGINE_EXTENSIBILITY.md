# Reasoning Engine Extensibility Guide

**Building Custom Reasoning Components with Elegance**

*Every component is a protocol. Every extension is a possibility.*

---

## Philosophy

> **"Good frameworks are extended, not modified. Great frameworks make extension elegant."**

The Reasoning Engine is built on **protocols, not implementations**. Every component can be replaced, extended, or composed. This guide shows you how to build custom reasoning components that integrate seamlessly.

---

## Table of Contents

1. [Extension Points](#extension-points)
2. [Custom Reasoners](#custom-reasoners)
3. [Custom Verifiers](#custom-verifiers)
4. [Custom Planners](#custom-planners)
5. [Plugin Architecture](#plugin-architecture)
6. [Hook System](#hook-system)
7. [Recipes](#recipes)

---

## Extension Points

The Reasoning Engine has **7 extension points**:

```
┌─────────────────────────────────────────────────┐
│                                                 │
│              ReasoningEngine                    │
│                                                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. QueryPlanner      → Intent & Planning       │
│  2. ChainOfThought    → Evidence & Synthesis    │
│  3. SelfVerifier      → Verification Logic      │
│  4. Backtracker       → Contradiction Handling  │
│  5. ModeBandit        → Mode Selection          │
│  6. ProvenanceTracker → Scratchpad Integration  │
│  7. MetricsCollector  → Performance Tracking    │
│                                                 │
└─────────────────────────────────────────────────┘
```

Each component is **swappable** via dependency injection.

---

## Custom Reasoners

### Example 1: Domain-Specific Reasoner

**Use Case**: Legal document analysis with citations.

```python
from HoloLoom.reasoning.chain_of_thought import ChainOfThought
from HoloLoom.reasoning.types import ReasoningStep, StepType, Synthesis
import re

class LegalReasoner(ChainOfThought):
    """Chain-of-thought for legal document analysis."""

    def __init__(self):
        super().__init__()
        self.citation_pattern = r'\d+\s+\w+\.\s+\d+[a-z]?'  # e.g., "42 USC 1983"

    def generate_standard_chain(self, query, intent, context):
        """Legal reasoning with citations."""
        chain = []

        # Step 1: Extract legal citations
        citations = self._extract_citations(query.text, context)
        chain.append(ReasoningStep(
            thought=f"Legal analysis: {len(citations)} citations found",
            evidence=f"Primary authorities: {', '.join(citations[:3])}",
            confidence=0.95 if citations else 0.6,
            step_type=StepType.UNDERSTANDING
        ))

        # Step 2: Identify legal issues
        issues = self._identify_legal_issues(query.text)
        chain.append(ReasoningStep(
            thought=f"Legal issues: {', '.join(issues)}",
            evidence=f"{len(issues)} distinct issues identified",
            confidence=0.9,
            step_type=StepType.EVIDENCE
        ))

        # Step 3: Apply legal standards
        for issue in issues:
            standard = self._find_applicable_standard(issue, citations)
            chain.append(ReasoningStep(
                thought=f"Standard for {issue}: {standard['rule']}",
                evidence=f"Citing: {standard['citation']}",
                confidence=standard['confidence'],
                step_type=StepType.SYNTHESIS
            ))

        # Step 4: Legal conclusion
        conclusion = self._synthesize_legal_conclusion(issues, citations)
        chain.append(ReasoningStep(
            thought=conclusion['conclusion'],
            evidence=f"Based on {len(citations)} authorities",
            confidence=conclusion['confidence'],
            step_type=StepType.SYNTHESIS
        ))

        return chain

    def _extract_citations(self, text, context):
        """Extract legal citations from text and context."""
        citations = []

        # From query
        citations.extend(re.findall(self.citation_pattern, text))

        # From context shards
        for shard in context.shards:
            citations.extend(re.findall(self.citation_pattern, shard.text))

        return list(set(citations))  # Deduplicate

    def _identify_legal_issues(self, text):
        """Identify legal issues from text."""
        issue_keywords = {
            'liability': ['negligence', 'duty', 'breach', 'damages'],
            'constitutional': ['first amendment', 'due process', 'equal protection'],
            'contract': ['breach', 'performance', 'consideration'],
            'criminal': ['intent', 'mens rea', 'actus reus'],
        }

        identified = []
        text_lower = text.lower()

        for issue_type, keywords in issue_keywords.items():
            if any(kw in text_lower for kw in keywords):
                identified.append(issue_type)

        return identified

    def _find_applicable_standard(self, issue, citations):
        """Find legal standard for issue."""
        # Simplified - in production, would query legal database
        standards = {
            'liability': {
                'rule': 'Duty + Breach + Causation + Damages',
                'citation': citations[0] if citations else 'Common Law',
                'confidence': 0.9
            },
            'constitutional': {
                'rule': 'Strict Scrutiny Test',
                'citation': citations[0] if citations else 'Const. Art. I',
                'confidence': 0.85
            },
        }

        return standards.get(issue, {
            'rule': 'Case-specific analysis required',
            'citation': 'N/A',
            'confidence': 0.5
        })

    def _synthesize_legal_conclusion(self, issues, citations):
        """Synthesize legal conclusion."""
        if len(citations) >= 3:
            return {
                'conclusion': f"Well-supported analysis across {len(issues)} legal issues",
                'confidence': 0.9
            }
        elif len(citations) >= 1:
            return {
                'conclusion': f"Preliminary analysis of {len(issues)} issues",
                'confidence': 0.75
            }
        else:
            return {
                'conclusion': "Analysis requires additional legal authority",
                'confidence': 0.5
            }
```

**Usage**:
```python
from HoloLoom.reasoning import ReasoningEngine

# Create engine with custom reasoner
engine = ReasoningEngine()
engine.cot_generator = LegalReasoner()  # Inject custom reasoner

# Now all reasoning uses legal-specific logic
result = await engine.reason(query, features, context)
```

---

### Example 2: Multi-Language Reasoner

**Use Case**: Reasoning across multiple languages with translation.

```python
from HoloLoom.reasoning.chain_of_thought import ChainOfThought
from HoloLoom.reasoning.types import ReasoningStep, StepType

class MultilingualReasoner(ChainOfThought):
    """Reasoning with automatic translation."""

    def __init__(self, primary_language='en'):
        super().__init__()
        self.primary_language = primary_language
        self.translations = {}  # Cache

    def generate_standard_chain(self, query, intent, context):
        # Detect query language
        query_lang = self._detect_language(query.text)

        # Translate if needed
        if query_lang != self.primary_language:
            translated_query = self._translate(query.text, query_lang, self.primary_language)
            chain = [ReasoningStep(
                thought=f"Query language: {query_lang}, translated to {self.primary_language}",
                evidence=f"Original: {query.text[:50]}...\nTranslated: {translated_query[:50]}...",
                confidence=0.9,
                step_type=StepType.UNDERSTANDING
            )]
        else:
            chain = []
            translated_query = query.text

        # Reason in primary language
        primary_chain = super().generate_standard_chain(
            Query(text=translated_query),
            intent,
            context
        )
        chain.extend(primary_chain)

        # Translate conclusion back if needed
        if query_lang != self.primary_language:
            conclusion = primary_chain[-1].thought
            translated_conclusion = self._translate(conclusion, self.primary_language, query_lang)

            chain.append(ReasoningStep(
                thought=f"Conclusion (in {query_lang}): {translated_conclusion}",
                evidence=f"Original ({self.primary_language}): {conclusion}",
                confidence=0.85,
                step_type=StepType.SYNTHESIS
            ))

        return chain

    def _detect_language(self, text):
        """Detect language (placeholder - use real library)."""
        # In production: use langdetect or spacy
        return 'en'  # Default

    def _translate(self, text, from_lang, to_lang):
        """Translate text (placeholder - use real API)."""
        cache_key = f"{from_lang}:{to_lang}:{text}"

        if cache_key in self.translations:
            return self.translations[cache_key]

        # In production: use Google Translate API, DeepL, etc.
        translated = text  # Placeholder

        self.translations[cache_key] = translated
        return translated
```

---

## Custom Verifiers

### Example 3: Confidence Calibration Verifier

**Use Case**: Ensure reasoning confidence matches actual accuracy.

```python
from HoloLoom.reasoning.verifier import SelfVerifier
from HoloLoom.reasoning.types import VerificationResult, VerificationSeverity

class CalibratedVerifier(SelfVerifier):
    """Verifier that calibrates confidence scores."""

    def __init__(self, threshold=0.75, calibration_data=None):
        super().__init__(threshold)
        # Historical data: predicted confidence → actual accuracy
        self.calibration_data = calibration_data or {}

    async def verify(self, chain, context):
        # Run base verification
        base_result = await super().verify(chain, context)

        # Calculate raw confidence
        raw_confidence = sum(s.confidence for s in chain) / len(chain)

        # Calibrate based on historical data
        calibrated_confidence = self._calibrate(raw_confidence)

        # Adjust result
        if calibrated_confidence < self.threshold:
            return VerificationResult(
                passed=False,
                issue=f"Calibrated confidence ({calibrated_confidence:.2f}) below threshold",
                correction="Confidence overestimated based on historical accuracy",
                severity=VerificationSeverity.WARNING,
                confidence=calibrated_confidence
            )

        return VerificationResult(
            passed=True,
            confidence=calibrated_confidence
        )

    def _calibrate(self, raw_confidence):
        """Calibrate confidence based on historical accuracy."""
        # Simple binning approach
        confidence_bins = {
            (0.0, 0.5): lambda c: c * 0.8,   # Low conf → reduce further
            (0.5, 0.7): lambda c: c * 0.9,   # Medium → slight reduction
            (0.7, 0.85): lambda c: c,        # Good → no change
            (0.85, 1.0): lambda c: c * 1.05, # High → slight boost
        }

        for (low, high), calibration_fn in confidence_bins.items():
            if low <= raw_confidence < high:
                return min(1.0, calibration_fn(raw_confidence))

        return raw_confidence

    def update_calibration(self, predicted_confidence, actual_accuracy):
        """Update calibration data with new observation."""
        # Bin the confidence
        bin_key = round(predicted_confidence, 1)

        if bin_key not in self.calibration_data:
            self.calibration_data[bin_key] = []

        self.calibration_data[bin_key].append(actual_accuracy)

        # Update calibration function based on running average
        # (In production: use isotonic regression or Platt scaling)
```

---

### Example 4: Fact-Checking Verifier

**Use Case**: Verify claims against a knowledge base.

```python
from HoloLoom.reasoning.verifier import SelfVerifier
from HoloLoom.reasoning.types import VerificationResult, VerificationSeverity

class FactCheckingVerifier(SelfVerifier):
    """Verifier that fact-checks claims."""

    def __init__(self, knowledge_base, threshold=0.75):
        super().__init__(threshold)
        self.kb = knowledge_base  # External knowledge base API

    async def verify(self, chain, context):
        # Run base verification
        base_result = await super().verify(chain, context)

        # Extract factual claims
        claims = self._extract_claims(chain)

        # Fact-check each claim
        fact_check_results = []
        for claim in claims:
            result = await self._fact_check(claim)
            fact_check_results.append(result)

        # Analyze results
        false_claims = [r for r in fact_check_results if not r['verified']]

        if false_claims:
            return VerificationResult(
                passed=False,
                issue=f"{len(false_claims)} unverified claims found",
                correction=f"False claims: {[c['claim'] for c in false_claims]}",
                severity=VerificationSeverity.CRITICAL,
                confidence=0.9
            )

        # All claims verified
        return VerificationResult(
            passed=True,
            confidence=0.95
        )

    def _extract_claims(self, chain):
        """Extract factual claims from reasoning chain."""
        claims = []

        claim_indicators = ['is', 'are', 'was', 'were', 'has', 'have']

        for step in chain:
            # Simple heuristic: sentences with claim indicators
            sentences = step.thought.split('.')
            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in claim_indicators):
                    claims.append(sentence.strip())

        return claims

    async def _fact_check(self, claim):
        """Fact-check a claim against knowledge base."""
        # Query knowledge base
        kb_result = await self.kb.query(claim)

        # Determine if verified
        verified = kb_result['confidence'] >= 0.8 and kb_result['matches'] > 0

        return {
            'claim': claim,
            'verified': verified,
            'confidence': kb_result['confidence'],
            'sources': kb_result.get('sources', [])
        }
```

---

## Custom Planners

### Example 5: Hierarchical Task Planner

**Use Case**: Decompose complex tasks into hierarchical sub-tasks.

```python
from HoloLoom.reasoning.planner import QueryPlanner
from HoloLoom.reasoning.types import QueryPlan, PlanStep

class HierarchicalPlanner(QueryPlanner):
    """Planner that creates hierarchical task decomposition."""

    def create_plan(self, query, features, context):
        # Analyze query complexity
        intent = self.analyze_intent(query, features)

        if intent.complexity < 0.5:
            # Simple query → flat plan
            return super().create_plan(query, features, context)

        # Complex query → hierarchical plan
        return self._create_hierarchical_plan(query, intent)

    def _create_hierarchical_plan(self, query, intent):
        """Create hierarchical task decomposition."""
        # Level 1: High-level goals
        high_level_goals = self._identify_high_level_goals(query.text)

        all_steps = []
        dependencies = {}

        # Level 2: Decompose each goal into sub-tasks
        for i, goal in enumerate(high_level_goals):
            # Create goal step
            goal_step = PlanStep(
                question=f"High-level goal: {goal['description']}",
                required_for="Overall objective",
                complexity=goal['complexity']
            )
            all_steps.append(goal_step)
            goal_idx = len(all_steps) - 1

            # Create sub-tasks for this goal
            sub_tasks = self._decompose_goal(goal)
            for sub_task in sub_tasks:
                sub_step = PlanStep(
                    question=f"Sub-task: {sub_task['question']}",
                    required_for=goal['description'],
                    complexity=sub_task['complexity']
                )
                all_steps.append(sub_step)
                sub_idx = len(all_steps) - 1

                # Sub-task depends on goal
                dependencies[sub_idx] = [goal_idx]

        return QueryPlan(
            steps=all_steps,
            estimated_complexity=intent.complexity,
            dependencies=dependencies
        )

    def _identify_high_level_goals(self, text):
        """Identify high-level goals from query."""
        # Simple heuristic: split on "and", "also", "additionally"
        parts = text.split(' and ')

        goals = []
        for i, part in enumerate(parts):
            goals.append({
                'description': part.strip(),
                'complexity': 0.6 + (i * 0.1)  # Later goals slightly more complex
            })

        return goals

    def _decompose_goal(self, goal):
        """Decompose goal into sub-tasks."""
        # Generic decomposition
        return [
            {'question': f"Understand requirements for: {goal['description']}", 'complexity': 0.3},
            {'question': f"Gather information about: {goal['description']}", 'complexity': 0.5},
            {'question': f"Synthesize solution for: {goal['description']}", 'complexity': 0.7},
        ]
```

---

## Plugin Architecture

### Building Reasoning Plugins

**Pattern**: Composable plugins that extend engine behavior.

```python
from typing import Protocol, runtime_checkable
from HoloLoom.reasoning.types import ReasoningResult

@runtime_checkable
class ReasoningPlugin(Protocol):
    """Protocol for reasoning plugins."""

    async def before_reasoning(self, query, features, context):
        """Called before reasoning starts."""
        ...

    async def after_reasoning(self, result: ReasoningResult):
        """Called after reasoning completes."""
        ...

    async def on_step(self, step, step_index):
        """Called after each reasoning step."""
        ...


class LoggingPlugin:
    """Plugin that logs all reasoning steps."""

    def __init__(self, logger):
        self.logger = logger

    async def before_reasoning(self, query, features, context):
        self.logger.info(f"Starting reasoning for: {query.text}")

    async def after_reasoning(self, result):
        self.logger.info(
            f"Reasoning complete: {len(result.chain)} steps, "
            f"confidence={result.total_confidence:.2f}"
        )

    async def on_step(self, step, step_index):
        self.logger.debug(
            f"Step {step_index}: {step.step_type.value} "
            f"[{step.confidence:.2f}] {step.thought[:50]}"
        )


class MetricsPlugin:
    """Plugin that collects metrics."""

    def __init__(self):
        self.step_counts = {}
        self.total_duration = 0

    async def before_reasoning(self, query, features, context):
        self.start_time = time.time()

    async def after_reasoning(self, result):
        self.total_duration += result.duration_ms

        # Count step types
        for step in result.chain:
            step_type = step.step_type.value
            self.step_counts[step_type] = self.step_counts.get(step_type, 0) + 1

    async def on_step(self, step, step_index):
        pass  # No per-step metrics


class PluggableReasoningEngine:
    """Reasoning engine with plugin support."""

    def __init__(self, plugins=None):
        self.base_engine = ReasoningEngine()
        self.plugins = plugins or []

    def add_plugin(self, plugin):
        """Add a plugin."""
        self.plugins.append(plugin)

    async def reason(self, query, features, context):
        # Before hooks
        for plugin in self.plugins:
            await plugin.before_reasoning(query, features, context)

        # Run base reasoning
        result = await self.base_engine.reason(query, features, context)

        # Step hooks
        for i, step in enumerate(result.chain):
            for plugin in self.plugins:
                await plugin.on_step(step, i)

        # After hooks
        for plugin in self.plugins:
            await plugin.after_reasoning(result)

        return result
```

**Usage**:
```python
import logging

# Create plugins
logging_plugin = LoggingPlugin(logging.getLogger(__name__))
metrics_plugin = MetricsPlugin()

# Create engine with plugins
engine = PluggableReasoningEngine(plugins=[logging_plugin, metrics_plugin])

# Use normally
result = await engine.reason(query, features, context)

# Access plugin data
print(f"Total duration: {metrics_plugin.total_duration}ms")
print(f"Step counts: {metrics_plugin.step_counts}")
```

---

## Hook System

### Pre/Post Reasoning Hooks

```python
class HookableReasoningEngine:
    """Engine with hooks for pre/post processing."""

    def __init__(self):
        self.base_engine = ReasoningEngine()
        self.pre_hooks = []
        self.post_hooks = []

    def register_pre_hook(self, hook):
        """Register pre-reasoning hook.

        Hook signature: async def hook(query, features, context) -> (query, features, context)
        """
        self.pre_hooks.append(hook)

    def register_post_hook(self, hook):
        """Register post-reasoning hook.

        Hook signature: async def hook(result) -> result
        """
        self.post_hooks.append(hook)

    async def reason(self, query, features, context):
        # Pre-hooks
        for hook in self.pre_hooks:
            query, features, context = await hook(query, features, context)

        # Reasoning
        result = await self.base_engine.reason(query, features, context)

        # Post-hooks
        for hook in self.post_hooks:
            result = await hook(result)

        return result


# Example hooks
async def normalize_query_hook(query, features, context):
    """Pre-hook: Normalize query text."""
    query.text = query.text.lower().strip()
    return query, features, context


async def confidence_boost_hook(result):
    """Post-hook: Boost confidence if all steps agree."""
    if len(result.chain) >= 3:
        min_conf = min(s.confidence for s in result.chain)
        max_conf = max(s.confidence for s in result.chain)

        if max_conf - min_conf < 0.1:  # Steps agree
            result.total_confidence = min(1.0, result.total_confidence * 1.1)

    return result


# Usage
engine = HookableReasoningEngine()
engine.register_pre_hook(normalize_query_hook)
engine.register_post_hook(confidence_boost_hook)
```

---

## Recipes

### Recipe 1: Streaming Reasoning

**Use Case**: Stream reasoning steps as they're generated.

```python
from typing import AsyncIterator
from HoloLoom.reasoning.types import ReasoningStep

class StreamingReasoner:
    """Reasoner that yields steps as they're generated."""

    def __init__(self):
        self.base_engine = ReasoningEngine()

    async def reason_streaming(
        self,
        query,
        features,
        context
    ) -> AsyncIterator[ReasoningStep]:
        """Yield reasoning steps as they're generated."""

        # For now, generate all then yield
        # In production: truly stream from LLM
        result = await self.base_engine.reason(query, features, context)

        for step in result.chain:
            yield step
            await asyncio.sleep(0.1)  # Simulate streaming delay


# Usage
async def main():
    streamer = StreamingReasoner()

    async for step in streamer.reason_streaming(query, features, context):
        print(f"[{step.step_type.value}] {step.thought}")
        print(f"Confidence: {step.confidence:.2f}\n")
```

---

### Recipe 2: Retry with Backoff

**Use Case**: Retry reasoning on failure with exponential backoff.

```python
import asyncio
from typing import Optional

class RetryReasoner:
    """Reasoner with retry logic."""

    def __init__(self, max_retries=3, base_delay=1.0):
        self.base_engine = ReasoningEngine()
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def reason_with_retry(
        self,
        query,
        features,
        context,
        min_confidence=0.7
    ):
        """Reason with retry on low confidence."""

        for attempt in range(self.max_retries):
            try:
                result = await self.base_engine.reason(query, features, context)

                # Success if confidence meets threshold
                if result.total_confidence >= min_confidence:
                    return result

                # Low confidence → retry
                logger.warning(
                    f"Low confidence ({result.total_confidence:.2f}), "
                    f"retrying {attempt + 1}/{self.max_retries}"
                )

                # Exponential backoff
                delay = self.base_delay * (2 ** attempt)
                await asyncio.sleep(delay)

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.error(f"Reasoning failed: {e}, retrying...")
                await asyncio.sleep(self.base_delay * (2 ** attempt))

        # All retries exhausted
        logger.error("Max retries exhausted, returning best result")
        return result  # Return last result even if low confidence
```

---

### Recipe 3: Caching Reasoner

**Use Case**: Cache reasoning results for repeated queries.

```python
from functools import lru_cache
import hashlib
import json

class CachedReasoner:
    """Reasoner with result caching."""

    def __init__(self, cache_size=1000):
        self.base_engine = ReasoningEngine()
        self.cache = {}
        self.cache_size = cache_size

    def _cache_key(self, query, features, context):
        """Generate cache key."""
        key_data = {
            'query': query.text,
            'motifs': features.motifs[:5],  # First 5 motifs
            'shard_count': len(context.shards)
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    async def reason(self, query, features, context):
        """Reason with caching."""
        cache_key = self._cache_key(query, features, context)

        # Check cache
        if cache_key in self.cache:
            logger.info(f"Cache hit for query: {query.text[:50]}")
            return self.cache[cache_key]

        # Cache miss
        result = await self.base_engine.reason(query, features, context)

        # Store in cache
        if len(self.cache) >= self.cache_size:
            # Evict oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[cache_key] = result

        return result
```

---

## Summary

**Extension Points**: 7 components (planner, CoT, verifier, backtracker, bandit, provenance, metrics)

**Custom Components**:
- 2 custom reasoners (legal, multilingual)
- 2 custom verifiers (calibration, fact-checking)
- 1 custom planner (hierarchical)

**Plugin Architecture**:
- Protocol-based plugin system
- Logging, metrics, and custom plugins
- Hook system for pre/post processing

**Recipes**:
- Streaming reasoning
- Retry with backoff
- Result caching

---

**Next**: See examples in `demos/custom_reasoning_examples.py`
