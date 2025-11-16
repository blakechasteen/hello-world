# Reasoning Engine Extensibility Guide

**Protocol-Based Extension Without Complexity**

---

## Philosophy

> **"Good frameworks are extended, not modified. Great frameworks make extension elegant."**

The Reasoning Engine is built on protocols, not implementations. Every component can be replaced, extended, or composed.

**Key Principle**: Composition over inheritance. Dependency injection over tight coupling.

---

## Table of Contents

1. [Extension Points](#extension-points)
2. [Domain-Specific Reasoners](#domain-specific-reasoners)
3. [Custom Verifiers](#custom-verifiers)
4. [Composition Patterns](#composition-patterns)
5. [Testing Extensions](#testing-extensions)
6. [Plugin Architecture](#plugin-architecture)

---

## Extension Points

The Reasoning Engine has **7 extension points**:

```
ReasoningEngine
├── 1. QueryPlanner      → Intent analysis & planning
├── 2. ChainOfThought    → Evidence gathering & synthesis
├── 3. SelfVerifier      → Verification logic
├── 4. Backtracker       → Contradiction resolution
├── 5. ModeBandit        → Mode selection strategy
├── 6. ProvenanceTracker → Scratchpad integration
└── 7. MetricsCollector  → Performance tracking
```

Each component is **swappable** via dependency injection.

**Quick example**:
```python
from HoloLoom.reasoning.chain_of_thought import ChainOfThought
from HoloLoom.reasoning.engine import ReasoningEngine

class MyCustomReasoner(ChainOfThought):
    def generate_standard_chain(self, query, intent, context):
        return custom_chain

engine = ReasoningEngine()
engine.reasoner = MyCustomReasoner()
result = await engine.reason(query, features, context)
```

That's the pattern. Extend, inject, use.

---

## Domain-Specific Reasoners

### Example 1: Legal Reasoner

**Use case**: Legal document analysis with citations and standards.

```python
from HoloLoom.reasoning.chain_of_thought import ChainOfThought
from HoloLoom.reasoning.types import ReasoningStep, StepType
import re

class LegalReasoner(ChainOfThought):
    def __init__(self):
        super().__init__()
        self.citation_pattern = r'\d+\s+\w+\.\s+\d+[a-z]?'

    def generate_standard_chain(self, query, intent, context):
        chain = []

        citations = self._extract_citations(query.text, context)
        chain.append(ReasoningStep(
            thought=f"Legal analysis: {len(citations)} citations found",
            evidence=f"Primary authorities: {', '.join(citations[:3])}",
            confidence=0.95 if citations else 0.6,
            step_type=StepType.UNDERSTANDING
        ))

        issues = self._identify_legal_issues(query.text)
        chain.append(ReasoningStep(
            thought=f"Legal issues: {', '.join(issues)}",
            evidence=f"{len(issues)} distinct issues identified",
            confidence=0.9,
            step_type=StepType.EVIDENCE
        ))

        for issue in issues:
            standard = self._find_applicable_standard(issue, citations)
            chain.append(ReasoningStep(
                thought=f"Standard for {issue}: {standard['rule']}",
                evidence=f"Citing: {standard['citation']}",
                confidence=standard['confidence'],
                step_type=StepType.SYNTHESIS
            ))

        conclusion = self._synthesize_legal_conclusion(issues, citations)
        chain.append(ReasoningStep(
            thought=conclusion['conclusion'],
            evidence=f"Based on {len(citations)} authorities",
            confidence=conclusion['confidence'],
            step_type=StepType.SYNTHESIS
        ))

        return chain

    def _extract_citations(self, text, context):
        citations = re.findall(self.citation_pattern, text)
        context_citations = []
        for item in context:
            context_citations.extend(re.findall(self.citation_pattern, item.text))
        return list(set(citations + context_citations))

    def _identify_legal_issues(self, text):
        issue_keywords = ['liability', 'negligence', 'contract', 'tort', 'damages']
        return [kw for kw in issue_keywords if kw in text.lower()]

    def _find_applicable_standard(self, issue, citations):
        return {
            'rule': f"Standard rule for {issue}",
            'citation': citations[0] if citations else "N/A",
            'confidence': 0.85
        }

    def _synthesize_legal_conclusion(self, issues, citations):
        return {
            'conclusion': f"Based on {len(issues)} issues and {len(citations)} authorities",
            'confidence': min(0.9, 0.7 + 0.1 * len(citations))
        }
```

**Usage**:
```python
engine = ReasoningEngine()
engine.reasoner = LegalReasoner()
result = await engine.reason(legal_query, features, context)
```

---

### Example 2: Multilingual Reasoner

**Use case**: Cross-language reasoning with translation verification.

```python
from HoloLoom.reasoning.chain_of_thought import ChainOfThought
from HoloLoom.reasoning.types import ReasoningStep, StepType

class MultilingualReasoner(ChainOfThought):
    def __init__(self, translator):
        super().__init__()
        self.translator = translator

    def generate_standard_chain(self, query, intent, context):
        chain = []

        detected_lang = self._detect_language(query.text)
        chain.append(ReasoningStep(
            thought=f"Query language: {detected_lang}",
            evidence=f"Detected with {self.translator.confidence:.2f} confidence",
            confidence=self.translator.confidence,
            step_type=StepType.UNDERSTANDING
        ))

        if detected_lang != 'en':
            translated = self.translator.translate(query.text, target='en')
            chain.append(ReasoningStep(
                thought=f"Translated: {translated}",
                evidence=f"Original: {query.text}",
                confidence=0.85,
                step_type=StepType.UNDERSTANDING
            ))

            back_translated = self.translator.translate(translated, target=detected_lang)
            consistency = self._measure_consistency(query.text, back_translated)
            chain.append(ReasoningStep(
                thought=f"Translation consistency: {consistency:.2f}",
                evidence=f"Back-translation matches original: {consistency > 0.9}",
                confidence=consistency,
                step_type=StepType.VERIFICATION
            ))

        reasoning = super().generate_standard_chain(query, intent, context)
        chain.extend(reasoning)

        return chain

    def _detect_language(self, text):
        return self.translator.detect(text)

    def _measure_consistency(self, original, back_translated):
        original_words = set(original.lower().split())
        back_words = set(back_translated.lower().split())
        overlap = len(original_words & back_words)
        return overlap / max(len(original_words), len(back_words))
```

---

## Custom Verifiers

### Example 3: Calibrated Verifier

**Use case**: Adjust confidence based on historical accuracy.

```python
from HoloLoom.reasoning.verifier import SelfVerifier
from HoloLoom.reasoning.types import VerificationResult, VerificationSeverity

class CalibratedVerifier(SelfVerifier):
    def __init__(self, threshold=0.75):
        super().__init__(threshold)
        self.calibration_data = {}

    async def verify(self, chain, context):
        base_result = await super().verify(chain, context)

        raw_confidence = sum(s.confidence for s in chain) / len(chain)
        calibrated_confidence = self._calibrate(raw_confidence)

        if calibrated_confidence < self.threshold:
            return VerificationResult(
                passed=False,
                issue=f"Calibrated confidence ({calibrated_confidence:.2f}) below threshold",
                correction="Confidence overestimated based on historical accuracy",
                severity=VerificationSeverity.WARNING,
                confidence=calibrated_confidence
            )

        return VerificationResult(passed=True, confidence=calibrated_confidence)

    def _calibrate(self, raw_confidence):
        bins = {
            (0.0, 0.5): lambda c: c * 0.8,
            (0.5, 0.7): lambda c: c * 0.9,
            (0.7, 0.85): lambda c: c,
            (0.85, 1.0): lambda c: c * 1.05,
        }

        for (low, high), fn in bins.items():
            if low <= raw_confidence < high:
                return min(1.0, fn(raw_confidence))

        return raw_confidence

    def update_calibration(self, predicted, actual):
        bin_key = round(predicted, 1)
        if bin_key not in self.calibration_data:
            self.calibration_data[bin_key] = []
        self.calibration_data[bin_key].append(actual)
```

---

### Example 4: Fact-Checking Verifier

**Use case**: Verify claims against external knowledge base.

```python
from HoloLoom.reasoning.verifier import SelfVerifier
from HoloLoom.reasoning.types import VerificationResult, VerificationSeverity

class FactCheckingVerifier(SelfVerifier):
    def __init__(self, knowledge_base, threshold=0.75):
        super().__init__(threshold)
        self.kb = knowledge_base

    async def verify(self, chain, context):
        base_result = await super().verify(chain, context)
        claims = self._extract_claims(chain)
        fact_check_results = await asyncio.gather(*[
            self._fact_check(claim) for claim in claims
        ])

        false_claims = [r for r in fact_check_results if not r['verified']]

        if false_claims:
            return VerificationResult(
                passed=False,
                issue=f"{len(false_claims)} unverified claims",
                correction=f"False: {[c['claim'] for c in false_claims]}",
                severity=VerificationSeverity.CRITICAL,
                confidence=0.9
            )

        return VerificationResult(passed=True, confidence=0.95)

    def _extract_claims(self, chain):
        claims = []
        indicators = ['is', 'are', 'was', 'were', 'has', 'have']

        for step in chain:
            for sentence in step.thought.split('.'):
                if any(ind in sentence.lower() for ind in indicators):
                    claims.append(sentence.strip())

        return claims

    async def _fact_check(self, claim):
        kb_result = await self.kb.query(claim)
        return {
            'claim': claim,
            'verified': kb_result['confidence'] >= 0.8,
            'confidence': kb_result['confidence'],
            'sources': kb_result.get('sources', [])
        }
```

---

## Composition Patterns

### Pattern 1: Chaining Verifiers

**Combine multiple verification strategies.**

```python
class ChainedVerifier(SelfVerifier):
    def __init__(self, verifiers):
        super().__init__()
        self.verifiers = verifiers

    async def verify(self, chain, context):
        for verifier in self.verifiers:
            result = await verifier.verify(chain, context)
            if not result.passed:
                return result
        return VerificationResult(passed=True, confidence=0.95)

# Usage
verifier = ChainedVerifier([
    SelfVerifier(threshold=0.75),
    CalibratedVerifier(),
    FactCheckingVerifier(kb)
])
```

---

### Pattern 2: Ensemble Reasoners

**Combine multiple reasoning strategies.**

```python
class EnsembleReasoner(ChainOfThought):
    def __init__(self, reasoners):
        super().__init__()
        self.reasoners = reasoners

    async def generate_standard_chain(self, query, intent, context):
        chains = await asyncio.gather(*[
            reasoner.generate_standard_chain(query, intent, context)
            for reasoner in self.reasoners
        ])

        best_chain = max(chains, key=lambda c: sum(s.confidence for s in c) / len(c))
        return best_chain

# Usage
reasoner = EnsembleReasoner([
    LegalReasoner(),
    MultilingualReasoner(translator),
    ChainOfThought()
])
```

---

### Pattern 3: Conditional Reasoning

**Choose reasoner based on query characteristics.**

```python
class ConditionalReasoner(ChainOfThought):
    def __init__(self, reasoners):
        super().__init__()
        self.reasoners = reasoners

    async def generate_standard_chain(self, query, intent, context):
        for condition, reasoner in self.reasoners:
            if condition(query, intent):
                return await reasoner.generate_standard_chain(query, intent, context)

        return await super().generate_standard_chain(query, intent, context)

# Usage
reasoner = ConditionalReasoner([
    (lambda q, i: 'legal' in q.text.lower(), LegalReasoner()),
    (lambda q, i: self.detect_language(q.text) != 'en', MultilingualReasoner(translator))
])
```

---

## Testing Extensions

### Unit Testing Custom Components

```python
import pytest
from HoloLoom.reasoning.types import Query, Features, Context, QueryIntent, QueryType

@pytest.mark.asyncio
async def test_legal_reasoner():
    reasoner = LegalReasoner()
    query = Query(text="Analyze 42 USC 1983 liability")
    intent = QueryIntent(
        type=QueryType.ANALYTICAL,
        requirements=['legal analysis'],
        key_concepts=['liability', '42 USC 1983'],
        complexity=0.7,
        confidence=0.9
    )
    context = [Context(text="42 USC 1983 provides civil rights remedy")]

    chain = reasoner.generate_standard_chain(query, intent, context)

    assert len(chain) >= 3
    assert any('42 USC 1983' in step.thought for step in chain)
    assert all(step.confidence > 0.0 for step in chain)

@pytest.mark.asyncio
async def test_calibrated_verifier():
    verifier = CalibratedVerifier(threshold=0.75)

    high_conf_chain = [
        ReasoningStep("High conf", "Evidence", 0.95, StepType.SYNTHESIS),
        ReasoningStep("High conf", "Evidence", 0.90, StepType.SYNTHESIS),
    ]

    result = await verifier.verify(high_conf_chain, [])
    assert result.passed
    assert result.confidence >= 0.75

    low_conf_chain = [
        ReasoningStep("Low conf", "Evidence", 0.4, StepType.SYNTHESIS),
    ]

    result = await verifier.verify(low_conf_chain, [])
    assert not result.passed
```

---

### Integration Testing

```python
@pytest.mark.asyncio
async def test_legal_reasoner_integration():
    engine = ReasoningEngine()
    engine.reasoner = LegalReasoner()

    query = Query(text="What is the standard for 42 USC 1983 claims?")
    features = Features(motifs=['liability', 'civil rights'])
    context = [Context(text="42 USC 1983 addresses civil rights violations")]

    result = await engine.reason(query, features, context)

    assert result.total_confidence > 0.7
    assert len(result.chain) >= 3
    assert result.mode == ReasoningMode.STANDARD
```

---

## Plugin Architecture

### Creating Plugins

```python
from typing import Protocol

class ReasoningPlugin(Protocol):
    async def before_reasoning(self, query, features, context):
        ...

    async def after_reasoning(self, result):
        ...

    async def on_step(self, step, index):
        ...

class LoggingPlugin:
    async def before_reasoning(self, query, features, context):
        logger.info(f"Starting reasoning: {query.text[:50]}")

    async def after_reasoning(self, result):
        logger.info(f"Completed: {result.total_confidence:.2f}")

    async def on_step(self, step, index):
        logger.debug(f"Step {index}: [{step.confidence:.2f}] {step.thought}")

class MetricsPlugin:
    def __init__(self):
        self.start_time = None

    async def before_reasoning(self, query, features, context):
        self.start_time = time.time()

    async def after_reasoning(self, result):
        duration = (time.time() - self.start_time) * 1000
        metrics.record_duration(result.mode.value, duration)
        metrics.record_confidence(result.total_confidence)
```

---

### Pluggable Engine

```python
class PluggableReasoningEngine(ReasoningEngine):
    def __init__(self, mode=ReasoningMode.STANDARD, plugins=None):
        super().__init__(mode)
        self.plugins = plugins or []

    async def reason(self, query, features, context, mode=None):
        for plugin in self.plugins:
            await plugin.before_reasoning(query, features, context)

        result = await super().reason(query, features, context, mode)

        for plugin in self.plugins:
            await plugin.after_reasoning(result)

        return result

# Usage
engine = PluggableReasoningEngine(plugins=[
    LoggingPlugin(),
    MetricsPlugin()
])

result = await engine.reason(query, features, context)
```

---

## Best Practices

### 1. Prefer Composition Over Inheritance

```python
# Good: Composition
class MyReasoner:
    def __init__(self):
        self.legal = LegalReasoner()
        self.multilingual = MultilingualReasoner(translator)

    async def reason(self, query, features, context):
        if 'legal' in query.text:
            return await self.legal.generate_standard_chain(query, intent, context)
        else:
            return await self.multilingual.generate_standard_chain(query, intent, context)

# Less flexible: Deep inheritance
class MyReasoner(LegalReasoner, MultilingualReasoner):
    pass
```

### 2. Use Dependency Injection

```python
# Good: Dependencies injected
class MyVerifier(SelfVerifier):
    def __init__(self, kb, threshold=0.75):
        super().__init__(threshold)
        self.kb = kb

# Less flexible: Hardcoded dependencies
class MyVerifier(SelfVerifier):
    def __init__(self):
        super().__init__()
        self.kb = KnowledgeBase("hardcoded-url")
```

### 3. Test Extensions Independently

```python
# Test custom reasoner without engine
async def test_reasoner():
    reasoner = LegalReasoner()
    chain = reasoner.generate_standard_chain(query, intent, context)
    assert len(chain) > 0

# Then test integration
async def test_integration():
    engine = ReasoningEngine()
    engine.reasoner = LegalReasoner()
    result = await engine.reason(query, features, context)
    assert result.total_confidence > 0.7
```

### 4. Document Extension Points

```python
class MyReasoner(ChainOfThought):
    """
    Custom reasoner for domain X.

    Extension points:
    - _extract_domain_features(): Extract domain-specific features
    - _apply_domain_rules(): Apply domain-specific rules
    - _synthesize_conclusion(): Create domain-specific conclusion

    Example:
        reasoner = MyReasoner()
        engine = ReasoningEngine()
        engine.reasoner = reasoner
        result = await engine.reason(query, features, context)
    """
    pass
```

---

## Extension Checklist

Before deploying custom extensions:

- [ ] Extends appropriate base class
- [ ] Implements required methods
- [ ] Unit tests cover key functionality
- [ ] Integration tests verify engine compatibility
- [ ] Documentation explains what, why, how
- [ ] Error handling for edge cases
- [ ] Performance profiled (no regressions)
- [ ] Composition tested (works with other extensions)

---

## Next Steps

1. **Start simple**: Extend one component (reasoner or verifier)
2. **Test thoroughly**: Unit tests + integration tests
3. **Compose carefully**: Combine extensions via composition patterns
4. **Monitor always**: Use plugins for logging and metrics
5. **Document clearly**: Future you will thank present you

---

**"Good frameworks are extended, not modified. Great frameworks make extension elegant."**
