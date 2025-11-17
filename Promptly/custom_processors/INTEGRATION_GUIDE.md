# Custom Processors Integration Guide

This guide explains how to integrate the custom processors into your Promptly workflows and existing applications.

## Table of Contents

1. [Installation](#installation)
2. [Registration](#registration)
3. [Configuration](#configuration)
4. [Usage Patterns](#usage-patterns)
5. [Advanced Features](#advanced-features)
6. [Performance Tuning](#performance-tuning)
7. [Error Handling](#error-handling)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Installation

### Requirements

```bash
# Core dependencies
pip install requests  # For API and Webhook processors

# Optional dependencies
pip install redis  # For Redis caching (future support)
pip install pytest  # For running tests
```

### Directory Structure

```
Promptly/
├── promptly/
│   └── plugins/
│       ├── base.py
│       └── __init__.py
└── custom_processors/
    ├── __init__.py
    ├── webhook.py
    ├── api.py
    ├── validation.py
    ├── cache.py
    ├── aggregation.py
    ├── README.md
    ├── INTEGRATION_GUIDE.md
    ├── examples/
    │   └── workflows/
    │       ├── cached_llm.yaml
    │       ├── multi_model_ensemble.yaml
    │       └── data_pipeline.yaml
    └── tests/
        └── test_all_processors.py
```

## Registration

### Method 1: Manual Registration

```python
from promptly.plugins import get_registry
from custom_processors import (
    WebhookProcessor,
    APIIntegrationProcessor,
    DataValidationProcessor,
    CacheProcessor,
    AggregationProcessor
)

# Get the global plugin registry
registry = get_registry()

# Register each processor
registry.register_chain_processor(WebhookProcessor)
registry.register_chain_processor(APIIntegrationProcessor)
registry.register_chain_processor(DataValidationProcessor)
registry.register_chain_processor(CacheProcessor)
registry.register_chain_processor(AggregationProcessor)

# Verify registration
processors = registry.list_chain_processors()
print("Registered processors:", [p['name'] for p in processors])
```

### Method 2: Automatic Registration

Add to your application startup:

```python
# app/startup.py
def register_custom_processors():
    """Register all custom processors on startup"""
    from promptly.plugins import get_registry
    import custom_processors

    registry = get_registry()

    # Auto-register all processors in the module
    for name in dir(custom_processors):
        obj = getattr(custom_processors, name)
        if (isinstance(obj, type) and
            hasattr(obj, 'process') and
            hasattr(obj, 'name')):
            try:
                registry.register_chain_processor(obj)
                print(f"Registered: {obj().name}")
            except Exception as e:
                print(f"Failed to register {name}: {e}")

# Call during app initialization
register_custom_processors()
```

### Method 3: Plugin Directory Loading

```python
from promptly.plugins import get_loader
from pathlib import Path

# Get plugin loader
loader = get_loader()

# Load from directory
custom_dir = Path(__file__).parent / "custom_processors"
loader.load_from_directory(custom_dir)
```

## Configuration

### Environment Variables

Create a `.env` file for sensitive configuration:

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
CUSTOM_API_KEY=your_key

# Webhook URLs
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
MONITORING_WEBHOOK_URL=https://your-monitor.com/webhook

# Webhook Secrets
WEBHOOK_SECRET=your_secret_key_here

# Cache Configuration
CACHE_DIR=.cache
CACHE_MAX_SIZE_MB=100
CACHE_DEFAULT_TTL=3600

# Rate Limiting
DEFAULT_RATE_LIMIT=10  # requests per second
BURST_SIZE=20
```

Load environment variables:

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Use in configurations
openai_key = os.getenv("OPENAI_API_KEY")
webhook_secret = os.getenv("WEBHOOK_SECRET")
```

### Configuration Files

Create `config.yaml` for non-sensitive settings:

```yaml
processors:
  api:
    default_timeout: 30
    default_retry_attempts: 3
    rate_limit:
      requests_per_second: 10
      burst_size: 20

  cache:
    strategy: multi_level
    memory:
      max_size: 1000
      default_ttl: 300
    disk:
      cache_dir: .cache
      max_size_mb: 100
      default_ttl: 3600

  webhook:
    default_timeout: 30
    retry_strategy: exponential
    max_attempts: 3

  validation:
    strict_mode: false
    on_error: log

  aggregation:
    default_strategy: voting
    confidence_threshold: 0.7
```

Load configuration:

```python
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Use in workflows
cache_config = config['processors']['cache']
```

## Usage Patterns

### Pattern 1: Caching Expensive API Calls

```python
from custom_processors import CacheProcessor, APIIntegrationProcessor

def cached_api_workflow(input_data):
    """Workflow with API caching"""

    cache = CacheProcessor()
    api = APIIntegrationProcessor()

    # 1. Check cache
    cache_config = {
        "operation": "get",
        "strategy": "multi_level",
        "levels": [
            {"type": "memory", "ttl": 300},
            {"type": "disk", "ttl": 3600}
        ],
        "key": {"fields": ["input"]}
    }

    result = cache.process(input_data, cache_config)

    # 2. On cache miss, call API
    if not result.get("cache_hit"):
        api_config = {
            "url": "https://api.example.com/endpoint",
            "auth": {"type": "bearer", "token": os.getenv("API_KEY")},
            "body": {"data": input_data["input"]}
        }

        result = api.process(input_data, api_config)

        # 3. Cache the result
        cache_config["operation"] = "set"
        cache.process(result, cache_config)

    return result
```

### Pattern 2: Multi-Model Consensus

```python
from custom_processors import AggregationProcessor, APIIntegrationProcessor

def multi_model_consensus(prompt):
    """Get consensus from multiple models"""

    api = APIIntegrationProcessor()
    aggregator = AggregationProcessor()

    # 1. Call multiple models
    models = [
        {"name": "gpt-3.5", "url": "...", "weight": 0.3},
        {"name": "claude", "url": "...", "weight": 0.4},
        {"name": "gpt-4", "url": "...", "weight": 0.3}
    ]

    responses = []
    for model in models:
        config = {
            "url": model["url"],
            "auth": {"type": "bearer", "token": os.getenv(f"{model['name'].upper()}_KEY")},
            "body": {"prompt": prompt}
        }
        result = api.process({"prompt": prompt}, config)
        responses.append({
            "value": result.get("api_response"),
            "source": model["name"],
            "weight": model["weight"]
        })

    # 2. Aggregate with weighted voting
    agg_config = {
        "strategy": "weighted_voting",
        "source_field": "responses",
        "weights": {m["name"]: m["weight"] for m in models},
        "confidence": {"enabled": True, "threshold": 0.7}
    }

    result = aggregator.process({"responses": responses}, agg_config)

    return {
        "consensus": result.get("aggregated_result"),
        "confidence": result.get("aggregation_metadata", {}).get("confidence_score"),
        "individual_responses": responses
    }
```

### Pattern 3: Data Validation Pipeline

```python
from custom_processors import DataValidationProcessor, WebhookProcessor

def validate_and_notify(user_data):
    """Validate data and send notifications"""

    validator = DataValidationProcessor()
    webhook = WebhookProcessor()

    # 1. Comprehensive validation
    validation_config = {
        "validations": [
            # Schema validation
            {
                "type": "schema",
                "schema": {
                    "type": "object",
                    "required": ["email", "name"],
                    "properties": {
                        "email": {"type": "string", "pattern": "^[^@]+@[^@]+$"},
                        "name": {"type": "string", "minLength": 1}
                    }
                }
            },
            # PII detection
            {
                "type": "pii",
                "detect": ["email", "phone", "ssn"],
                "action": "flag"
            },
            # Sanitization
            {
                "type": "sanitize",
                "rules": [
                    {"type": "xss_filter"},
                    {"type": "html_escape"}
                ]
            }
        ],
        "on_validation_error": "log"
    }

    result = validator.process({"user_data": user_data}, validation_config)

    # 2. Send notification based on validation
    webhook_config = {
        "url": os.getenv("NOTIFICATION_WEBHOOK"),
        "body": {
            "event": "validation_completed",
            "valid": result.get("is_valid"),
            "pii_found": len(result.get("pii_detections", [])),
            "timestamp": str(datetime.now())
        }
    }

    webhook.process(result, webhook_config)

    return result
```

### Pattern 4: Rate-Limited API with Fallback

```python
from custom_processors import APIIntegrationProcessor

def rate_limited_api_call(request_data):
    """API call with rate limiting and fallback"""

    api = APIIntegrationProcessor()

    config = {
        "url": "https://api.example.com/endpoint",
        "auth": {"type": "api_key", "key": "X-API-Key", "value": os.getenv("API_KEY")},
        "rate_limit": {
            "enabled": True,
            "requests_per_second": 5,
            "burst_size": 10
        },
        "retry": {
            "enabled": True,
            "max_attempts": 3,
            "backoff_strategy": "exponential"
        },
        "fallback": {
            "enabled": True,
            "value": "Default response when API fails"
        },
        "cache": {
            "enabled": True,
            "ttl": 300
        }
    }

    result = api.process(request_data, config)

    return {
        "response": result.get("api_response"),
        "fallback_used": result.get("api_fallback_used", False),
        "from_cache": result.get("api_cached", False)
    }
```

## Advanced Features

### Custom Validators

```python
from custom_processors import DataValidationProcessor

# Create validator
validator = DataValidationProcessor()

# Register custom validator function
def validate_business_hours(data, context):
    """Custom validator: Check if timestamp is during business hours"""
    from datetime import datetime

    timestamp = data.get("timestamp")
    if not timestamp:
        return False

    dt = datetime.fromisoformat(timestamp)
    return 9 <= dt.hour <= 17  # 9 AM to 5 PM

validator.register_validator("business_hours", validate_business_hours)

# Use in configuration
config = {
    "validations": [
        {
            "type": "business_rule",
            "rules": [
                {
                    "name": "business_hours",
                    "message": "Request must be during business hours"
                }
            ]
        }
    ]
}
```

### Custom Cache Key Generators

```python
from custom_processors import CacheProcessor
import hashlib

cache = CacheProcessor()

def semantic_cache_key(data, config):
    """Generate cache key based on semantic similarity"""
    text = data.get("input", "")

    # Simple example: hash normalized text
    normalized = text.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()

cache.register_key_generator("semantic", semantic_cache_key)

# Use custom key generator
config = {
    "key": {
        "generator": "semantic",
        "prefix": "semantic_cache"
    }
}
```

### Custom Aggregation Functions

```python
from custom_processors import AggregationProcessor

aggregator = AggregationProcessor()

def custom_ensemble(results, config):
    """Custom ensemble method with confidence weighting"""
    from collections import Counter

    # Weight results by their confidence scores
    weighted_votes = []
    for result in results:
        confidence = result.get("confidence", 1.0)
        value = result.get("value")

        # Add weighted votes
        weight = int(confidence * 10)
        weighted_votes.extend([value] * weight)

    # Return most common
    counter = Counter(weighted_votes)
    return counter.most_common(1)[0][0] if counter else None

aggregator.register_aggregator("confidence_ensemble", custom_ensemble)

# Use custom aggregator
config = {
    "strategy": "confidence_ensemble",
    "source_field": "model_outputs"
}
```

## Performance Tuning

### Caching Strategy

```python
# For frequently accessed, small data
config = {
    "strategy": "memory",
    "max_size": 1000,
    "ttl": 300  # 5 minutes
}

# For infrequently accessed, large data
config = {
    "strategy": "disk",
    "cache_dir": ".cache",
    "max_size_mb": 500,
    "ttl": 86400  # 24 hours
}

# For best performance
config = {
    "strategy": "multi_level",
    "levels": [
        {"type": "memory", "max_size": 100, "ttl": 300},   # Hot cache
        {"type": "disk", "max_size_mb": 1000, "ttl": 3600}  # Warm cache
    ]
}
```

### Rate Limiting

```python
# Conservative (prevent throttling)
config = {
    "rate_limit": {
        "requests_per_second": 5,
        "burst_size": 10
    }
}

# Aggressive (maximize throughput)
config = {
    "rate_limit": {
        "requests_per_second": 50,
        "burst_size": 100
    }
}

# Balanced
config = {
    "rate_limit": {
        "requests_per_second": 10,
        "burst_size": 20
    }
}
```

### Retry Strategy

```python
# Fast failure (time-sensitive)
config = {
    "retry": {
        "enabled": True,
        "max_attempts": 2,
        "backoff_strategy": "constant",
        "initial_delay": 0.5
    }
}

# Resilient (critical operations)
config = {
    "retry": {
        "enabled": True,
        "max_attempts": 5,
        "backoff_strategy": "exponential",
        "initial_delay": 1.0,
        "max_delay": 30.0
    }
}
```

## Error Handling

### Graceful Degradation

```python
def resilient_workflow(input_data):
    """Workflow with comprehensive error handling"""

    try:
        # Try primary path
        result = primary_processor.process(input_data, config)

        if result.get("success"):
            return result

        # Try fallback
        return fallback_processor.process(input_data, fallback_config)

    except Exception as e:
        # Log error
        logger.error(f"Workflow failed: {e}")

        # Return safe default
        return {
            "success": False,
            "error": str(e),
            "fallback_data": get_default_response()
        }
```

### Error Monitoring

```python
from custom_processors import WebhookProcessor

def monitor_errors(error, context):
    """Send error notifications"""

    webhook = WebhookProcessor()

    config = {
        "url": os.getenv("ERROR_WEBHOOK"),
        "body": {
            "level": "error",
            "message": str(error),
            "context": context,
            "timestamp": str(datetime.now())
        },
        "retry": {
            "enabled": True,
            "max_attempts": 3
        }
    }

    webhook.process({"error": str(error)}, config)

# Use in try/except blocks
try:
    result = processor.process(data, config)
except Exception as e:
    monitor_errors(e, {"processor": "api", "data": data})
```

## Best Practices

### 1. Security

```python
# ✓ Good: Store secrets in environment variables
api_key = os.getenv("API_KEY")

# ✗ Bad: Hardcode secrets
api_key = "sk-1234567890"  # DON'T DO THIS

# ✓ Good: Validate and sanitize all input
validator.process(user_input, validation_config)

# ✓ Good: Use HMAC signatures for webhooks
webhook_config = {
    "signature": {
        "enabled": True,
        "secret": os.getenv("WEBHOOK_SECRET"),
        "algorithm": "sha256"
    }
}

# ✓ Good: Redact PII before logging
validation_config = {
    "validations": [
        {
            "type": "pii",
            "detect": ["email", "ssn", "credit_card"],
            "action": "redact"
        }
    ]
}
```

### 2. Performance

```python
# ✓ Good: Cache expensive operations
cache_config = {"enabled": True, "ttl": 3600}

# ✓ Good: Use rate limiting to avoid throttling
rate_limit_config = {"enabled": True, "requests_per_second": 10}

# ✓ Good: Enable multi-level caching
cache_config = {
    "strategy": "multi_level",
    "levels": [
        {"type": "memory", "ttl": 300},
        {"type": "disk", "ttl": 3600}
    ]
}

# ✓ Good: Monitor cache hit rates
metrics = cache.get_metrics()
if metrics["hit_rate"] < 0.5:
    print("Consider increasing cache TTL")
```

### 3. Reliability

```python
# ✓ Good: Always configure retry logic
retry_config = {
    "enabled": True,
    "max_attempts": 3,
    "backoff_strategy": "exponential"
}

# ✓ Good: Implement fallback strategies
fallback_config = {
    "enabled": True,
    "value": "default_response"
}

# ✓ Good: Validate outputs
output_validation = {
    "validations": [
        {"type": "schema", "schema": output_schema},
        {"type": "data_quality", "checks": [{"type": "completeness"}]}
    ]
}

# ✓ Good: Monitor and alert on failures
webhook.process(result, notification_config)
```

### 4. Maintainability

```python
# ✓ Good: Use configuration files
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# ✓ Good: Centralize processor creation
def create_api_processor():
    processor = APIIntegrationProcessor()
    # Add custom configuration
    return processor

# ✓ Good: Document configuration
"""
Configuration:
- cache.ttl: Cache time-to-live in seconds (default: 300)
- retry.max_attempts: Maximum retry attempts (default: 3)
"""
```

## Troubleshooting

### Common Issues

#### 1. Cache Not Working

```python
# Check cache is enabled
assert config["cache"]["enabled"] == True

# Check TTL is set
assert config["cache"]["ttl"] > 0

# Check cache directory exists
import os
cache_dir = config["cache"].get("cache_dir", ".cache")
os.makedirs(cache_dir, exist_ok=True)

# Check metrics
metrics = cache.get_metrics()
print(f"Hit rate: {metrics['hit_rate']}")
```

#### 2. Rate Limiting Too Aggressive

```python
# Increase limits
config["rate_limit"]["requests_per_second"] = 20
config["rate_limit"]["burst_size"] = 40

# Or disable temporarily
config["rate_limit"]["enabled"] = False
```

#### 3. Webhook Failing

```python
# Test webhook URL
import requests
response = requests.post(webhook_url, json={"test": True})
print(f"Status: {response.status_code}")

# Check signature configuration
assert config["signature"]["secret"] is not None

# Increase timeout
config["timeout"] = 60.0

# Check retry configuration
config["retry"]["enabled"] = True
config["retry"]["max_attempts"] = 5
```

#### 4. Validation Too Strict

```python
# Disable strict mode
config["strict"] = False

# Change error handling
config["on_validation_error"] = "log"  # Instead of "throw"

# Adjust thresholds
config["validations"][0]["checks"][0]["threshold"] = 0.7  # Lower threshold
```

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Inspect processor state
print(f"Cache size: {cache._get_memory_cache(config).size()}")
print(f"Rate limiter tokens: {api._rate_limiters['default'].tokens}")

# Check configuration
import json
print(json.dumps(config, indent=2))

# Test individual components
result = processor.pre_process(input_data)
print(f"Pre-processed: {result}")

result = processor.process(input_data, config)
print(f"Processed: {result}")

result = processor.post_process(result)
print(f"Post-processed: {result}")
```

## Support

For issues and questions:
- **GitHub Issues**: https://github.com/your-repo/issues
- **Documentation**: See README.md for detailed API reference
- **Examples**: Check `examples/workflows/` for working examples
- **Tests**: Run `tests/test_all_processors.py` for validation

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - See LICENSE file for details
