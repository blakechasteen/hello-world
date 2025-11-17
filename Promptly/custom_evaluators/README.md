# Custom Evaluator Examples for Promptly

Complete real-world examples demonstrating how to extend Promptly with custom evaluators for specialized use cases.

## Overview

This package contains 5 production-ready custom evaluator implementations:

1. **Business Logic Evaluator** - Industry-specific validation and scoring
2. **Multi-Model Consensus Evaluator** - LLM ensemble evaluation with disagreement detection
3. **Structured Output Evaluator** - Schema validation for JSON/XML/YAML/SQL
4. **Domain-Specific Evaluator** - Medical/Legal/Code/Marketing specialized evaluators
5. **Human-in-the-Loop Evaluator** - Queue-based human review workflows

All evaluators follow the `BaseEvaluator` protocol and include:
- ✅ Complete working implementations
- ✅ Comprehensive docstrings and type hints
- ✅ Configuration examples
- ✅ Test cases
- ✅ Real-world use case documentation

## Installation

### Dependencies

```bash
# Core dependencies (required)
pip install promptly

# Optional dependencies for specific evaluators
pip install jsonschema  # For JSON schema validation
pip install pyyaml      # For YAML validation
pip install sqlparse    # For SQL validation
pip install lxml        # For XML validation
pip install redis       # For Redis-based HITL queue
pip install openai      # For OpenAI consensus evaluation
pip install anthropic   # For Anthropic consensus evaluation
```

### Installation

```bash
# Clone or copy the custom_evaluators directory
cd Promptly
python -m pip install -e .
```

## Quick Start

```python
from custom_evaluators import (
    CustomerServiceEvaluator,
    MultiModelConsensusEvaluator,
    JSONValidator,
    CodeSecurityEvaluator,
    HumanInTheLoopEvaluator
)

# 1. Business Logic - Customer Service
cs_eval = CustomerServiceEvaluator(min_length=50, max_length=500)
score = cs_eval.evaluate(actual="Hello! I'll help you...", expected="")
print(f"Customer service score: {score:.2f}")

# 2. Multi-Model Consensus
from custom_evaluators.consensus import MockProvider, ConsensusStrategy

providers = [
    MockProvider("model_a", bias=0.8),
    MockProvider("model_b", bias=0.7)
]
consensus_eval = MultiModelConsensusEvaluator(
    providers=providers,
    strategy=ConsensusStrategy.MEAN
)
score = consensus_eval.evaluate("output", "expected")

# 3. JSON Validation
json_schema = {
    "type": "object",
    "required": ["name", "email"],
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"}
    }
}
json_eval = JSONValidator(schema=json_schema)
score = json_eval.evaluate('{"name": "John", "email": "john@example.com"}', "")

# 4. Code Security
code_eval = CodeSecurityEvaluator(language="python")
score = code_eval.evaluate("def secure_function():\n    pass", "")

# 5. Human-in-the-Loop
hitl_eval = HumanInTheLoopEvaluator(auto_approve_threshold=0.9)
score = hitl_eval.evaluate("output", "expected", context={'pre_score': 0.95})
```

## Evaluator Reference

### 1. Business Logic Evaluator

Custom scoring based on business rules with configurable thresholds and weights.

#### CustomerServiceEvaluator

Validates customer service responses for quality and professionalism.

```python
from custom_evaluators import CustomerServiceEvaluator

evaluator = CustomerServiceEvaluator(
    min_length=50,   # Minimum response length
    max_length=500   # Maximum response length
)

response = """
Hello! Thank you for contacting us.
I understand your concern and will help resolve this immediately.
I'll follow up with you within 24 hours.
Best regards, Support Team
"""

metrics = evaluator.get_metrics(response, "")

print(f"Score: {metrics['score']:.2f}")
print(f"Passed: {metrics['passed']}")
print(f"Rules passed: {metrics['rules_passed']}/{metrics['total_rules']}")

# Output:
# Score: 0.85
# Passed: True
# Rules passed: 6/7
```

**Validation Rules:**
- ✓ Professional greeting
- ✓ Empathy expression
- ✓ Resolution offer
- ✓ Professional closing
- ✓ No dismissive language
- ✓ Appropriate length
- ✓ Clear next steps

#### FinancialComplianceEvaluator

Validates financial content for regulatory compliance.

```python
from custom_evaluators import FinancialComplianceEvaluator

evaluator = FinancialComplianceEvaluator(jurisdiction="US")

compliant_text = """
Past performance does not guarantee future results. All investments carry
risk and you may lose money. This is not financial advice. Please consult
with a qualified financial advisor.
"""

metrics = evaluator.get_metrics(compliant_text, "")
print(f"Compliance score: {metrics['score']:.2f}")
print(f"Rule violations: {len(metrics.get('errors', []))}")
```

**Compliance Checks:**
- ✓ Risk disclosures
- ✓ No guaranteed return claims
- ✓ Professional advice disclaimer
- ✓ No insider trading references
- ✓ Financial data accuracy

#### MedicalAccuracyEvaluator

Validates medical content for safety and accuracy.

```python
from custom_evaluators import MedicalAccuracyEvaluator

evaluator = MedicalAccuracyEvaluator(require_disclaimer=True)

safe_text = """
These symptoms could indicate various conditions. Please consult your doctor
or healthcare provider for proper evaluation. This is not medical advice.
If experiencing chest pain, call 911 immediately.
"""

metrics = evaluator.get_metrics(safe_text, "")
print(f"Safety score: {metrics['score']:.2f}")
```

**Safety Checks:**
- ✓ Medical disclaimer present
- ✓ Contraindication warnings
- ✓ No definitive diagnoses
- ✓ No prescription recommendations
- ✓ Emergency situation handling

### 2. Multi-Model Consensus Evaluator

Aggregates scores from multiple LLM providers with disagreement detection.

```python
from custom_evaluators.consensus import (
    MultiModelConsensusEvaluator,
    ConsensusStrategy,
    OpenAIProvider,
    AnthropicProvider,
    MockProvider
)

# Option 1: Use real LLM providers
providers = [
    OpenAIProvider(model="gpt-4", api_key="your-key"),
    AnthropicProvider(model="claude-3-sonnet-20240229", api_key="your-key")
]

# Option 2: Use mock providers for testing
providers = [
    MockProvider("model_a", bias=0.8),
    MockProvider("model_b", bias=0.7),
    MockProvider("model_c", bias=0.6)
]

evaluator = MultiModelConsensusEvaluator(
    providers=providers,
    strategy=ConsensusStrategy.WEIGHTED,
    weights={"model_a": 2.0, "model_b": 1.5, "model_c": 1.0},
    disagreement_threshold=0.3,  # Flag if std dev > 0.3
    cache_enabled=True,
    cache_ttl=3600  # 1 hour
)

metrics = evaluator.get_metrics(
    actual="This is a comprehensive response.",
    expected="A good response.",
    context={'evaluation_prompt': 'Evaluate clarity and completeness.'}
)

print(f"Consensus score: {metrics['score']:.3f}")
print(f"Individual scores: {metrics['individual_scores']}")
print(f"Disagreement: {metrics['disagreement']:.3f}")
print(f"Flagged for review: {metrics['flagged']}")
```

**Consensus Strategies:**

- `MEAN` - Simple average of all scores
- `MEDIAN` - Median score (robust to outliers)
- `WEIGHTED` - Weighted average based on model reliability
- `MAJORITY_VOTE` - Binary voting (pass/fail)
- `MIN` - Most conservative (lowest score)
- `MAX` - Most optimistic (highest score)
- `UNANIMOUS` - All models must agree

**Features:**
- ✓ Multiple LLM provider support
- ✓ Configurable consensus strategies
- ✓ Disagreement detection and flagging
- ✓ Response caching for cost optimization
- ✓ Detailed metrics for debugging

### 3. Structured Output Evaluator

Validates structured formats with schema compliance.

#### JSONValidator

```python
from custom_evaluators import JSONValidator

schema = {
    "type": "object",
    "required": ["user", "timestamp"],
    "properties": {
        "user": {
            "type": "object",
            "required": ["name", "email"],
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"}
            }
        },
        "timestamp": {"type": "string"}
    }
}

validator = JSONValidator(
    schema=schema,
    required_fields=["user.name", "user.email"],
    strict=True
)

json_output = '''{
    "user": {
        "name": "John Doe",
        "email": "john@example.com"
    },
    "timestamp": "2024-01-15T10:30:00Z"
}'''

metrics = validator.get_metrics(json_output, "")

print(f"Score: {metrics['score']:.2f}")
print(f"Valid JSON: {metrics['is_valid_json']}")
print(f"Valid Schema: {metrics['is_valid_schema']}")
print(f"Missing fields: {metrics['missing_fields']}")
```

#### XMLValidator

```python
from custom_evaluators import XMLValidator

validator = XMLValidator(
    required_elements=["title", "author", "content"]
)

xml_output = """
<document>
    <title>Example Document</title>
    <author>John Doe</author>
    <content>Document content here</content>
</document>
"""

metrics = validator.get_metrics(xml_output, "")
print(f"Well-formed: {metrics['is_well_formed']}")
```

#### SQLValidator

```python
from custom_evaluators import SQLValidator

validator = SQLValidator(
    allowed_operations={'SELECT', 'INSERT', 'UPDATE'},
    forbidden_operations={'DROP', 'DELETE', 'TRUNCATE'}
)

safe_query = "SELECT * FROM users WHERE id = ?"
unsafe_query = "DROP TABLE users; --"

print(f"Safe query score: {validator.evaluate(safe_query, '')}")
print(f"Unsafe query score: {validator.evaluate(unsafe_query, '')}")
```

#### YAMLValidator

```python
from custom_evaluators import YAMLValidator

validator = YAMLValidator(
    required_keys=['name', 'version', 'dependencies']
)

yaml_output = """
name: my-application
version: 1.0.0
dependencies:
  - package1>=1.0
  - package2>=2.0
"""

metrics = validator.get_metrics(yaml_output, "")
print(f"Valid YAML: {metrics['is_valid_yaml']}")
```

### 4. Domain-Specific Evaluators

Specialized evaluators for specific industries and use cases.

#### MedicalTerminologyEvaluator

```python
from custom_evaluators.domain_specific import MedicalTerminologyEvaluator

evaluator = MedicalTerminologyEvaluator()

professional_text = """
The patient presents with acute dyspnea and tachycardia. Pulmonary
examination reveals decreased breath sounds bilaterally. Recommend
immediate cardiology consultation.
"""

metrics = evaluator.get_metrics(professional_text, "")

print(f"Professional score: {metrics['score']:.2f}")
print(f"Medical terms found: {metrics['terminology']['term_count']}")
print(f"Professional language: {metrics['professionalism']['is_professional']}")
```

#### LegalCitationEvaluator

```python
from custom_evaluators.domain_specific import LegalCitationEvaluator

evaluator = LegalCitationEvaluator(citation_style="bluebook")

legal_text = """
Introduction: The plaintiff brings this motion under 42 USC § 1983.

Analysis: In Miranda v. Arizona, 384 U.S. 436 (1966), the Supreme Court
established that suspects must be informed of their rights.

Conclusion: Therefore, the evidence should be suppressed.
"""

metrics = evaluator.get_metrics(legal_text, "")

print(f"Legal quality: {metrics['score']:.2f}")
print(f"Citations found: {metrics['citations']['total_citations']}")
print(f"Well-structured: {metrics['formatting']['is_well_structured']}")
```

#### CodeSecurityEvaluator

```python
from custom_evaluators.domain_specific import CodeSecurityEvaluator

evaluator = CodeSecurityEvaluator(language="python")

secure_code = '''
def get_user(user_id: int) -> dict:
    """Retrieve user information."""
    try:
        query = "SELECT * FROM users WHERE id = ?"
        cursor.execute(query, (user_id,))
        return cursor.fetchone()
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
'''

metrics = evaluator.get_metrics(secure_code, "")

print(f"Security score: {metrics['score']:.2f}")
print(f"Vulnerabilities: {metrics['security']['vulnerability_count']}")
print(f"Has documentation: {metrics['best_practices']['has_documentation']}")
```

**Security Checks:**
- SQL injection vulnerabilities
- XSS vulnerabilities
- Hardcoded secrets
- Insecure random number generation
- Best practices compliance

#### BrandVoiceEvaluator

```python
from custom_evaluators.domain_specific import BrandVoiceEvaluator

brand_guidelines = {
    'voice_attributes': ['friendly', 'professional', 'helpful'],
    'key_phrases': ['we', 'our customers', 'together', 'support'],
    'avoid_phrases': ['cheap', 'expensive', 'complicated'],
    'tone': 'conversational',
    'formality': 'semi-formal'
}

evaluator = BrandVoiceEvaluator(brand_guidelines=brand_guidelines)

marketing_copy = """
We're excited to help you find the perfect solution! Our team is here
to support you every step of the way. Together, we'll make this happen.
"""

metrics = evaluator.get_metrics(marketing_copy, "")

print(f"Brand alignment: {metrics['score']:.2f}")
print(f"Key phrases used: {metrics['phrases']['key_phrase_count']}")
print(f"Tone match: {metrics['tone']['tone_score']:.2f}")
```

### 5. Human-in-the-Loop Evaluator

Queue-based human review with inter-annotator agreement tracking.

```python
from custom_evaluators.hitl import (
    HumanInTheLoopEvaluator,
    ReviewQueue,
    ReviewStatus,
    InterAnnotatorAgreement
)

# Initialize review queue
queue = ReviewQueue(
    backend="sqlite",  # or "redis", "file"
    config={'db_path': './reviews.db'}
)

# Create evaluator
evaluator = HumanInTheLoopEvaluator(
    queue=queue,
    auto_approve_threshold=0.9,  # Auto-approve if pre_score >= 0.9
    auto_reject_threshold=0.3    # Auto-reject if pre_score <= 0.3
)

# Evaluate with automatic approval
high_quality = "Excellent, comprehensive response with all details."
score = evaluator.evaluate(
    actual=high_quality,
    expected="A good response",
    context={'pre_score': 0.95}
)
print(f"Auto-approved: {score}")  # 0.95

# Evaluate requiring human review
medium_quality = "Decent response, could be better."
score = evaluator.evaluate(
    actual=medium_quality,
    expected="A good response",
    context={'pre_score': 0.6}
)
print(f"Pending review: {score}")  # 0.5

# Get pending reviews
pending = evaluator.get_pending_reviews(limit=10)
print(f"Items waiting for review: {len(pending)}")

# Human reviews an item
item = pending[0]
evaluator.review(
    item_id=item.id,
    score=0.75,
    feedback="Good response, minor improvements needed",
    reviewer="reviewer@example.com"
)

# Check review status
status = evaluator.get_review_status(item.id)
print(f"Review status: {status['status']}")
print(f"Score: {status['score']}")

# Calculate inter-annotator agreement
reviewer1_ratings = [5, 4, 3, 5, 4]
reviewer2_ratings = [5, 4, 4, 5, 4]

agreement = InterAnnotatorAgreement()
percentage = agreement.percentage_agreement(reviewer1_ratings, reviewer2_ratings)
kappa = agreement.cohens_kappa(reviewer1_ratings, reviewer2_ratings)

print(f"Agreement: {percentage:.1%}")
print(f"Cohen's Kappa: {kappa:.3f}")
```

**Queue Backends:**

- **SQLite** - Local database (default)
- **Redis** - Distributed queue for multiple reviewers
- **File** - Simple JSON file storage

**Features:**
- ✓ Persistent review queue
- ✓ Auto-approve/reject based on thresholds
- ✓ Batch review workflows
- ✓ Inter-annotator agreement metrics
- ✓ Review status tracking

## Integration with Promptly

### Using Custom Evaluators with Promptly

```python
from promptly import Promptly
from custom_evaluators import CustomerServiceEvaluator

# Initialize Promptly
promptly = Promptly()

# Create custom evaluator
cs_evaluator = CustomerServiceEvaluator()

# Define evaluation function for Promptly
def evaluate_customer_service(output, expected, context=None):
    metrics = cs_evaluator.get_metrics(output, expected, context)
    return {
        'score': metrics['score'],
        'passed': metrics['passed'],
        'details': metrics
    }

# Use in prompt evaluation
test_cases = [
    {
        'inputs': {'customer_query': 'My order is late'},
        'expected': 'Professional empathetic response',
        'evaluator': evaluate_customer_service
    }
]

results = promptly.eval_prompt('customer_service_prompt', test_cases)

for result in results:
    print(f"Score: {result['score']:.2f}")
    print(f"Passed: {result['passed']}")
```

### Combining Multiple Evaluators

```python
from custom_evaluators import (
    CustomerServiceEvaluator,
    BrandVoiceEvaluator,
    JSONValidator
)

class MultiEvaluator:
    """Combine multiple evaluators with weights"""

    def __init__(self):
        self.evaluators = {
            'customer_service': (CustomerServiceEvaluator(), 0.4),
            'brand_voice': (BrandVoiceEvaluator(), 0.3),
            'json_structure': (JSONValidator(required_fields=['response']), 0.3)
        }

    def evaluate(self, actual, expected, context=None):
        total_score = 0.0
        total_weight = 0.0
        details = {}

        for name, (evaluator, weight) in self.evaluators.items():
            score = evaluator.evaluate(actual, expected, context)
            total_score += score * weight
            total_weight += weight
            details[name] = score

        final_score = total_score / total_weight

        return {
            'score': final_score,
            'details': details
        }

# Usage
multi_eval = MultiEvaluator()
result = multi_eval.evaluate('{"response": "Hello! I can help."}', "")
print(f"Combined score: {result['score']:.2f}")
print(f"Details: {result['details']}")
```

## Advanced Use Cases

### 1. A/B Testing with Consensus Evaluation

```python
from custom_evaluators.consensus import (
    MultiModelConsensusEvaluator,
    ConsensusStrategy,
    MockProvider
)

providers = [MockProvider(f"model_{i}", bias=0.7+i*0.05) for i in range(3)]

evaluator = MultiModelConsensusEvaluator(
    providers=providers,
    strategy=ConsensusStrategy.MEAN,
    disagreement_threshold=0.2
)

# Test variant A vs variant B
variant_a = "Concise response."
variant_b = "Detailed comprehensive response with examples."

score_a = evaluator.evaluate(variant_a, "expected")
score_b = evaluator.evaluate(variant_b, "expected")

print(f"Variant A: {score_a:.3f}")
print(f"Variant B: {score_b:.3f}")
print(f"Winner: {'A' if score_a > score_b else 'B'}")
```

### 2. Compliance Pipeline

```python
from custom_evaluators import (
    FinancialComplianceEvaluator,
    JSONValidator,
    HumanInTheLoopEvaluator
)

class CompliancePipeline:
    """Multi-stage compliance validation"""

    def __init__(self):
        self.compliance_eval = FinancialComplianceEvaluator()
        self.structure_eval = JSONValidator(required_fields=['disclosure'])
        self.hitl_eval = HumanInTheLoopEvaluator(auto_approve_threshold=0.95)

    def validate(self, content):
        # Stage 1: Structure validation
        structure_score = self.structure_eval.evaluate(content, "")
        if structure_score < 0.8:
            return {'passed': False, 'reason': 'Invalid structure'}

        # Stage 2: Compliance check
        compliance_score = self.compliance_eval.evaluate(content, "")
        if compliance_score < 0.9:
            return {'passed': False, 'reason': 'Compliance violation'}

        # Stage 3: Human review for edge cases
        final_score = self.hitl_eval.evaluate(
            content, "",
            context={'pre_score': compliance_score}
        )

        return {
            'passed': final_score >= 0.9,
            'score': final_score,
            'requires_review': final_score == 0.5
        }

pipeline = CompliancePipeline()
result = pipeline.validate('{"disclosure": "Past performance..."}')
print(result)
```

### 3. Progressive Quality Gating

```python
from custom_evaluators import (
    JSONValidator,
    CodeSecurityEvaluator,
    HumanInTheLoopEvaluator
)

class QualityGate:
    """Progressive quality checks with escalating review"""

    def __init__(self):
        self.gates = [
            ('structure', JSONValidator(), 0.8),
            ('security', CodeSecurityEvaluator(), 0.9),
            ('human_review', HumanInTheLoopEvaluator(), 0.95)
        ]

    def check(self, code):
        for gate_name, evaluator, threshold in self.gates:
            score = evaluator.evaluate(code, "")

            if score < threshold:
                return {
                    'passed': False,
                    'failed_at': gate_name,
                    'score': score,
                    'threshold': threshold
                }

        return {'passed': True, 'score': 1.0}

gate = QualityGate()
result = gate.check('{"code": "def secure_func(): pass"}')
print(result)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python custom_evaluators/test_evaluators.py

# Run specific test class
python -m unittest custom_evaluators.test_evaluators.TestBusinessLogicEvaluators

# Run with verbose output
python custom_evaluators/test_evaluators.py -v
```

## Configuration Examples

### Environment Variables

```bash
# OpenAI API Key
export OPENAI_API_KEY=your-key-here

# Anthropic API Key
export ANTHROPIC_API_KEY=your-key-here

# Redis configuration
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

### Configuration File (YAML)

```yaml
# evaluators_config.yaml

customer_service:
  min_length: 50
  max_length: 500
  min_passing_score: 0.75

financial_compliance:
  jurisdiction: US
  min_passing_score: 0.9

consensus:
  strategy: weighted
  providers:
    - name: gpt-4
      weight: 2.0
    - name: claude-3
      weight: 1.5
  disagreement_threshold: 0.3
  cache_ttl: 3600

hitl:
  backend: redis
  auto_approve_threshold: 0.9
  auto_reject_threshold: 0.3
```

## Performance Considerations

### Caching

Enable caching for expensive operations:

```python
# Consensus evaluator with caching
evaluator = MultiModelConsensusEvaluator(
    cache_enabled=True,
    cache_ttl=3600  # 1 hour
)

# First call: actual LLM API calls
score1 = evaluator.evaluate("test", "expected")

# Second call: cached result (instant)
score2 = evaluator.evaluate("test", "expected")
```

### Batch Processing

Process multiple items efficiently:

```python
from custom_evaluators import CustomerServiceEvaluator

evaluator = CustomerServiceEvaluator()

responses = [
    "Response 1...",
    "Response 2...",
    "Response 3..."
]

# Batch evaluate
scores = [evaluator.evaluate(r, "") for r in responses]
print(f"Average score: {sum(scores)/len(scores):.2f}")
```

## Troubleshooting

### Common Issues

**Issue: "jsonschema not found"**
```bash
pip install jsonschema
```

**Issue: "Redis connection failed"**
```python
# Fallback to file backend
queue = ReviewQueue(backend="file")
```

**Issue: "OpenAI API rate limit"**
```python
# Use caching and reduce concurrent calls
evaluator = MultiModelConsensusEvaluator(
    cache_enabled=True,
    providers=[OpenAIProvider(model="gpt-3.5-turbo")]  # Use cheaper model
)
```

## Contributing

To add a new custom evaluator:

1. Inherit from `BaseEvaluator`
2. Implement `evaluate()` method
3. Implement `get_metrics()` method (optional)
4. Add tests to `test_evaluators.py`
5. Update `__init__.py` and this README

Example template:

```python
from promptly.plugins.base import BaseEvaluator

class MyCustomEvaluator(BaseEvaluator):
    def __init__(self, param1, param2):
        super().__init__(
            name="my_custom",
            description="My custom evaluator"
        )
        self.param1 = param1
        self.param2 = param2

    def evaluate(self, actual, expected, context=None):
        # Your evaluation logic
        score = 0.0
        # Calculate score...
        return score

    def get_metrics(self, actual, expected, context=None):
        return {
            'score': self.evaluate(actual, expected, context),
            # Additional metrics...
        }
```

## License

MIT License - see LICENSE file for details.

## Support

For issues, questions, or contributions:
- GitHub Issues: [link]
- Documentation: [link]
- Examples: See `examples/` directory

## Changelog

### Version 1.0.0 (2024-01)
- Initial release
- 5 complete evaluator implementations
- Comprehensive test suite
- Full documentation
