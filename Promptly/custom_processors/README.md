# Custom Chain Processors for Workflow Extension

This package provides 5 practical custom processor implementations for extending Promptly workflows with advanced capabilities.

## Processors Overview

### 1. **WebhookProcessor** (`webhook.py`)
Send HTTP webhooks during chain execution with retry logic and signature verification.

**Features:**
- Support for GET/POST/PUT/PATCH/DELETE methods
- Custom headers and query parameters
- Exponential backoff retry logic
- HMAC signature verification (SHA256/SHA1/MD5)
- Response validation
- Template variable substitution

**Use Cases:**
- Slack/Discord notifications
- Event logging to external systems
- Triggering external workflows
- Real-time monitoring

### 2. **APIIntegrationProcessor** (`api.py`)
Call external APIs (REST, GraphQL) with authentication, rate limiting, and caching.

**Features:**
- REST and GraphQL support
- Authentication: API key, Bearer, Basic, OAuth2, JWT
- Rate limiting (token bucket algorithm)
- Response caching with TTL
- Retry with exponential backoff
- Response mapping
- Fallback strategies

**Use Cases:**
- OpenAI/Anthropic API calls
- External data enrichment
- Third-party service integration
- Multi-model orchestration

### 3. **DataValidationProcessor** (`validation.py`)
Comprehensive schema validation, data sanitization, and PII detection.

**Features:**
- JSON Schema validation
- Data sanitization (HTML, SQL, XSS)
- PII detection and redaction (email, phone, SSN, credit card)
- Business rule validation
- Data quality checks
- Custom validators

**Use Cases:**
- Form validation
- Input sanitization
- Compliance (GDPR, HIPAA)
- Data quality assurance

### 4. **CacheProcessor** (`cache.py`)
Multi-level caching with memory, disk backends, and comprehensive metrics.

**Features:**
- Multi-level caching (L1 memory → L2 disk)
- Cache key generation strategies
- TTL and automatic eviction
- Hit/miss metrics
- Cost savings tracking
- Cache warming

**Use Cases:**
- LLM response caching
- Expensive computation caching
- Cost reduction
- Performance optimization

### 5. **AggregationProcessor** (`aggregation.py`)
Aggregate results from parallel branches with voting and statistical methods.

**Features:**
- Statistical aggregation (mean, median, mode)
- Voting mechanisms (majority, plurality, weighted)
- Consensus building
- Outlier detection (IQR, Z-score)
- Confidence scoring
- Quality filtering

**Use Cases:**
- Multi-model consensus
- Ensemble methods
- A/B testing
- Result validation

## Installation

```bash
# Install dependencies
pip install requests

# Optional for advanced features
pip install redis  # For Redis caching (future)
```

## Quick Start

### Example 1: Slack Notification Webhook

```python
from custom_processors.webhook import WebhookProcessor, create_slack_webhook

# Create processor
webhook = WebhookProcessor()

# Configure Slack webhook
config = create_slack_webhook(
    webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    channel="#notifications"
)

# Add retry and signature
config.update({
    "retry": {
        "enabled": True,
        "max_attempts": 3,
        "backoff_strategy": "exponential"
    }
})

# Process
result = webhook.process(
    step_input={"output": "Workflow completed successfully!"},
    step_config=config
)
```

### Example 2: OpenAI API Call with Caching

```python
from custom_processors.api import APIIntegrationProcessor, create_openai_config
from custom_processors.cache import CacheProcessor

# Create processors
api = APIIntegrationProcessor()
cache = CacheProcessor()

# Configure OpenAI API
api_config = create_openai_config(
    api_key="sk-...",
    model="gpt-3.5-turbo"
)

# Configure cache
cache_config = {
    "type": "cache",
    "operation": "get",
    "strategy": "multi_level",
    "levels": [
        {"type": "memory", "max_size": 100, "ttl": 300},
        {"type": "disk", "cache_dir": ".cache/llm", "ttl": 3600}
    ],
    "key": {
        "fields": ["input"],
        "prefix": "openai_cache",
        "version": "v1"
    }
}

# Try cache first
step_input = {"input": "What is AI?"}
cache_result = cache.process(step_input, cache_config)

if not cache_result.get("cache_hit"):
    # Cache miss - call API
    api_result = api.process(step_input, api_config)

    # Cache the result
    cache_config["operation"] = "set"
    cache.process(api_result, cache_config)

    print(f"API Response: {api_result['api_response']}")
else:
    print(f"Cached Response: {cache_result['cached_value']}")

# View metrics
metrics = cache.get_metrics()
print(f"Cache Hit Rate: {metrics['hit_rate']:.2%}")
print(f"Cost Savings: ${cache.get_cost_savings()['total_cost_saved']:.2f}")
```

### Example 3: Multi-Model Consensus

```python
from custom_processors.aggregation import AggregationProcessor

# Create processor
aggregator = AggregationProcessor()

# Simulate parallel model outputs
parallel_results = [
    "Paris is the capital of France.",
    "The capital of France is Paris.",
    "Paris",
    "Lyon",  # Outlier
    "Paris is the French capital."
]

# Configure consensus aggregation
config = {
    "type": "aggregation",
    "strategy": "voting",
    "source_field": "parallel_results",
    "target_field": "consensus_result",
    "voting": {
        "method": "majority",
        "threshold": 0.6
    },
    "statistical": {
        "detect_outliers": True,
        "outlier_method": "iqr",
        "remove_outliers": True
    },
    "confidence": {
        "enabled": True,
        "method": "agreement"
    }
}

# Aggregate
result = aggregator.process(
    step_input={"parallel_results": parallel_results},
    step_config=config
)

print(f"Consensus: {result['consensus_result']}")
print(f"Confidence: {result['aggregation_metadata']['confidence_score']:.2%}")
print(f"Outliers: {result['aggregation_metadata']['outliers']}")
```

### Example 4: Data Validation Pipeline

```python
from custom_processors.validation import DataValidationProcessor

# Create processor
validator = DataValidationProcessor()

# User registration data
user_data = {
    "name": "John Doe",
    "email": "john@example.com",
    "age": 25,
    "ssn": "123-45-6789",
    "comment": "<script>alert('xss')</script>Normal comment"
}

# Configure validation
config = {
    "type": "validation",
    "validations": [
        {
            "type": "schema",
            "schema": {
                "type": "object",
                "required": ["name", "email"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "email": {"type": "string", "pattern": "^[^@]+@[^@]+$"},
                    "age": {"type": "integer", "minimum": 18, "maximum": 150}
                }
            }
        },
        {
            "type": "sanitize",
            "rules": [
                {"type": "xss_filter"},
                {"type": "html_escape"}
            ]
        },
        {
            "type": "pii",
            "detect": ["email", "ssn"],
            "action": "redact",
            "replacement": "[REDACTED]"
        }
    ],
    "source_field": "user_data",
    "on_validation_error": "log"
}

# Validate
result = validator.process(
    step_input={"user_data": user_data},
    step_config=config
)

print(f"Valid: {result['is_valid']}")
print(f"Sanitized Data: {result['validated_data']}")
```

## Workflow Examples

### Workflow 1: LLM API with Caching and Monitoring

```yaml
name: cached_llm_workflow
version: 1.0

steps:
  # Step 1: Check cache
  - name: check_cache
    type: cache
    operation: get
    strategy: multi_level
    levels:
      - type: memory
        max_size: 100
        ttl: 300
      - type: disk
        cache_dir: .cache/llm
        ttl: 3600
    key:
      fields: [input, model]
      prefix: llm_cache
    on_miss: continue
    on_hit: return

  # Step 2: Call LLM API (only on cache miss)
  - name: call_llm
    type: api
    condition: "not check_cache.cache_hit"
    api_type: rest
    url: https://api.openai.com/v1/chat/completions
    method: POST
    auth:
      type: bearer
      token: ${OPENAI_API_KEY}
    body:
      model: "{model}"
      messages:
        - role: user
          content: "{input}"
    rate_limit:
      enabled: true
      requests_per_second: 3
      burst_size: 5
    retry:
      enabled: true
      max_attempts: 3
      backoff_strategy: exponential
    response_mapping:
      output: choices[0].message.content

  # Step 3: Cache the result
  - name: cache_result
    type: cache
    condition: "call_llm.api_success"
    operation: set
    strategy: multi_level
    levels:
      - type: memory
        ttl: 300
      - type: disk
        ttl: 3600

  # Step 4: Send webhook notification
  - name: notify
    type: webhook
    url: https://hooks.slack.com/services/YOUR/WEBHOOK
    method: POST
    body:
      text: "LLM request processed. Cache hit: {check_cache.cache_hit}"
    retry:
      enabled: true
      max_attempts: 2
```

### Workflow 2: Multi-Model Ensemble

```yaml
name: multi_model_ensemble
version: 1.0

steps:
  # Step 1: Validate input
  - name: validate_input
    type: validation
    validations:
      - type: schema
        schema:
          type: object
          required: [prompt]
          properties:
            prompt:
              type: string
              minLength: 1
              maxLength: 1000
      - type: sanitize
        rules:
          - type: xss_filter
          - type: normalize_whitespace

  # Step 2: Parallel model calls
  - name: parallel_models
    type: parallel
    tasks:
      - name: model_a
        type: api
        url: https://api.openai.com/v1/chat/completions
        auth:
          type: bearer
          token: ${OPENAI_KEY}
        body:
          model: gpt-3.5-turbo
          messages: [{role: user, content: "{prompt}"}]
        response_mapping:
          output: choices[0].message.content

      - name: model_b
        type: api
        url: https://api.anthropic.com/v1/messages
        auth:
          type: api_key
          key: x-api-key
          value: ${ANTHROPIC_KEY}
        body:
          model: claude-3-sonnet-20240229
          messages: [{role: user, content: "{prompt}"}]
        response_mapping:
          output: content[0].text

      - name: model_c
        type: api
        url: https://api.example.com/v1/generate
        auth:
          type: bearer
          token: ${CUSTOM_KEY}
        body:
          prompt: "{prompt}"
        response_mapping:
          output: response.text

  # Step 3: Aggregate results
  - name: aggregate
    type: aggregation
    strategy: weighted_voting
    source_field: parallel_models.results
    weights:
      model_a: 0.5
      model_b: 0.3
      model_c: 0.2
    statistical:
      detect_outliers: true
      outlier_method: iqr
    confidence:
      enabled: true
      method: agreement
      threshold: 0.7

  # Step 4: Validate output
  - name: validate_output
    type: validation
    validations:
      - type: data_quality
        checks:
          - type: completeness
            threshold: 0.9
      - type: pii
        detect: [email, phone, ssn]
        action: flag

  # Step 5: Notify on completion
  - name: notify
    type: webhook
    url: ${NOTIFICATION_URL}
    body:
      status: complete
      confidence: "{aggregate.confidence_score}"
      result: "{aggregate.consensus_result}"
```

### Workflow 3: Data Processing Pipeline

```yaml
name: data_processing_pipeline
version: 1.0

steps:
  # Step 1: Validate input data
  - name: validate
    type: validation
    validations:
      - type: schema
        schema:
          type: object
          required: [records]
          properties:
            records:
              type: array
              items:
                type: object
                required: [id, email, data]
      - type: pii
        detect: [email, phone, ssn, credit_card]
        action: redact
      - type: sanitize
        rules:
          - type: html_escape
          - type: sql_escape

  # Step 2: Enrich data via API
  - name: enrich
    type: loop
    loop_type: for_each
    items: "{validate.validated_data.records}"
    step:
      type: api
      url: https://api.enrichment.com/enrich
      auth:
        type: api_key
        key: X-API-Key
        value: ${ENRICHMENT_KEY}
      body:
        record: "{item}"
      cache:
        enabled: true
        ttl: 3600
        key_fields: [item.id]
      rate_limit:
        enabled: true
        requests_per_second: 10

  # Step 3: Aggregate results
  - name: aggregate
    type: aggregation
    strategy: merge
    source_field: enrich.results
    merge_strategy:
      conflicts: merge_nested
      nested: true

  # Step 4: Quality check
  - name: quality_check
    type: validation
    validations:
      - type: data_quality
        checks:
          - type: completeness
            threshold: 0.95
          - type: uniqueness
            fields: [id]
          - type: consistency

  # Step 5: Send to webhook
  - name: send_results
    type: webhook
    condition: "quality_check.is_valid"
    url: ${RESULTS_WEBHOOK}
    method: POST
    body:
      processed_count: "{aggregate.count}"
      quality_score: "{quality_check.quality_score}"
      data: "{aggregate.aggregated_result}"
    signature:
      enabled: true
      secret: ${WEBHOOK_SECRET}
      algorithm: sha256
    retry:
      enabled: true
      max_attempts: 3
      backoff_strategy: exponential
```

## Configuration Reference

### WebhookProcessor

```python
{
    "type": "webhook",
    "url": "https://api.example.com/webhook",
    "method": "POST",  # GET, POST, PUT, PATCH, DELETE
    "headers": {
        "Content-Type": "application/json",
        "Custom-Header": "value"
    },
    "body": {
        "message": "{output}",
        "timestamp": "{timestamp}"
    },
    "query_params": {"key": "value"},
    "timeout": 30.0,
    "retry": {
        "enabled": True,
        "max_attempts": 3,
        "backoff_strategy": "exponential",  # constant, linear, exponential, fibonacci, jitter
        "initial_delay": 1.0,
        "max_delay": 30.0,
        "multiplier": 2.0
    },
    "signature": {
        "enabled": True,
        "secret": "webhook_secret",
        "algorithm": "sha256",  # sha256, sha1, md5
        "header": "X-Webhook-Signature"
    },
    "response_validation": {
        "enabled": True,
        "expected_status": [200, 201, 202]
    }
}
```

### APIIntegrationProcessor

```python
{
    "type": "api",
    "api_type": "rest",  # rest, graphql
    "url": "https://api.example.com/endpoint",
    "method": "POST",
    "auth": {
        "type": "bearer",  # none, api_key, bearer, basic, oauth2, jwt
        "token": "your_token"
    },
    "headers": {"Content-Type": "application/json"},
    "body": {"data": "{input}"},
    "rate_limit": {
        "enabled": True,
        "requests_per_second": 10,
        "burst_size": 20
    },
    "cache": {
        "enabled": True,
        "ttl": 300,
        "key_fields": ["input"]
    },
    "retry": {
        "enabled": True,
        "max_attempts": 3,
        "backoff_strategy": "exponential"
    },
    "fallback": {
        "enabled": True,
        "value": "default response"
    },
    "response_mapping": {
        "output": "data.result",
        "status": "status"
    }
}
```

### DataValidationProcessor

```python
{
    "type": "validation",
    "validations": [
        {
            "type": "schema",
            "schema": {
                "type": "object",
                "required": ["field1"],
                "properties": {
                    "field1": {"type": "string"},
                    "field2": {"type": "integer", "minimum": 0}
                }
            }
        },
        {
            "type": "sanitize",
            "rules": [
                {"type": "html_escape"},
                {"type": "xss_filter"}
            ]
        },
        {
            "type": "pii",
            "detect": ["email", "phone", "ssn"],
            "action": "redact",  # redact, mask, remove, flag
            "replacement": "[REDACTED]"
        },
        {
            "type": "business_rule",
            "rules": [
                {
                    "name": "age_check",
                    "condition": "age >= 18",
                    "message": "Must be 18+"
                }
            ]
        }
    ],
    "on_validation_error": "throw"  # throw, log, continue
}
```

### CacheProcessor

```python
{
    "type": "cache",
    "operation": "get",  # get, set, delete, clear, warm
    "strategy": "multi_level",  # memory, disk, redis, multi_level
    "levels": [
        {
            "type": "memory",
            "max_size": 1000,
            "ttl": 300
        },
        {
            "type": "disk",
            "cache_dir": ".cache",
            "max_size_mb": 100,
            "ttl": 3600
        }
    ],
    "key": {
        "fields": ["input"],
        "prefix": "cache",
        "version": "v1"
    },
    "metrics": {
        "enabled": True,
        "track_cost": True,
        "cost_per_request": 0.01
    }
}
```

### AggregationProcessor

```python
{
    "type": "aggregation",
    "strategy": "voting",  # mean, median, mode, voting, consensus, best, merge
    "source_field": "parallel_results",
    "voting": {
        "method": "majority",  # majority, plurality, weighted
        "threshold": 0.6
    },
    "weights": {
        "source_1": 0.5,
        "source_2": 0.3,
        "source_3": 0.2
    },
    "statistical": {
        "detect_outliers": True,
        "outlier_method": "iqr",  # iqr, z_score
        "remove_outliers": True
    },
    "confidence": {
        "enabled": True,
        "method": "agreement",  # agreement, variance, entropy
        "threshold": 0.7
    },
    "quality_filter": {
        "enabled": True,
        "min_score": 0.5
    }
}
```

## Best Practices

### 1. Error Handling
- Always configure retry logic for external API calls
- Use fallback strategies for critical workflows
- Log errors with webhook notifications

### 2. Performance
- Enable caching for expensive operations
- Use rate limiting to avoid API throttling
- Implement multi-level caching for frequently accessed data

### 3. Security
- Always validate and sanitize user input
- Use HMAC signatures for webhooks
- Redact PII before logging or caching
- Store API keys in environment variables

### 4. Monitoring
- Enable metrics tracking on cache processors
- Send notifications on workflow completion/failure
- Track cost savings from caching

### 5. Testing
- Test with various input combinations
- Verify validation rules with edge cases
- Test retry logic with simulated failures
- Validate aggregation with outliers

## Integration with Promptly

### Register Processors

```python
from promptly.plugins import get_registry
from custom_processors import (
    WebhookProcessor,
    APIIntegrationProcessor,
    DataValidationProcessor,
    CacheProcessor,
    AggregationProcessor
)

# Get global registry
registry = get_registry()

# Register custom processors
registry.register_chain_processor(WebhookProcessor)
registry.register_chain_processor(APIIntegrationProcessor)
registry.register_chain_processor(DataValidationProcessor)
registry.register_chain_processor(CacheProcessor)
registry.register_chain_processor(AggregationProcessor)

# List available processors
processors = registry.list_chain_processors()
print(processors)
```

### Use in Chain Execution

```python
from promptly.chain_dsl import ChainDSL

# Create chain executor
dsl = ChainDSL()

# Load workflow
chain = dsl.load_chain("workflows/cached_llm_workflow.yaml")

# Execute
result = dsl.execute_chain(chain, {"input": "What is AI?"})
```

## Testing

See `tests/` directory for comprehensive integration tests.

```bash
# Run all tests
python -m pytest tests/

# Run specific processor tests
python -m pytest tests/test_webhook.py
python -m pytest tests/test_api.py
python -m pytest tests/test_validation.py
python -m pytest tests/test_cache.py
python -m pytest tests/test_aggregation.py
```

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- GitHub Issues: https://github.com/your-repo/issues
- Documentation: https://docs.your-site.com

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.
