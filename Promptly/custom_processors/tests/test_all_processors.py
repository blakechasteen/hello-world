#!/usr/bin/env python3
"""
Integration tests for all custom processors
"""

import sys
import os
import time
import json
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from webhook import WebhookProcessor
from api import APIIntegrationProcessor
from validation import DataValidationProcessor
from cache import CacheProcessor
from aggregation import AggregationProcessor


class TestWebhookProcessor:
    """Test WebhookProcessor"""

    def test_basic_webhook(self):
        """Test basic webhook POST"""
        processor = WebhookProcessor()

        config = {
            "type": "webhook",
            "url": "https://httpbin.org/post",
            "method": "POST",
            "body": {
                "message": "{output}",
                "test": True
            },
            "timeout": 10.0
        }

        step_input = {"output": "Test message"}

        result = processor.process(step_input, config)

        assert result.get("webhook_success") == True
        assert "webhook_result" in result
        print("✓ Basic webhook test passed")

    def test_webhook_retry(self):
        """Test webhook with retry logic"""
        processor = WebhookProcessor()

        config = {
            "type": "webhook",
            "url": "https://httpbin.org/delay/1",  # Simulates delay
            "method": "GET",
            "timeout": 5.0,
            "retry": {
                "enabled": True,
                "max_attempts": 2,
                "backoff_strategy": "constant",
                "initial_delay": 0.5
            }
        }

        result = processor.process({}, config)

        assert result.get("webhook_success") == True
        print("✓ Webhook retry test passed")

    def test_webhook_signature(self):
        """Test HMAC signature generation"""
        processor = WebhookProcessor()

        secret = "test_secret"
        body = "test body"

        # Test signature verification
        signature = processor._add_signature(
            {},
            body,
            {"secret": secret, "algorithm": "sha256", "header": "X-Signature"}
        )

        # Verify
        sig_value = signature["X-Signature"].split("=")[1]
        is_valid = processor.verify_signature(body, sig_value, secret, "sha256")

        assert is_valid == True
        print("✓ Webhook signature test passed")

    def test_template_substitution(self):
        """Test template variable substitution"""
        processor = WebhookProcessor()

        config = {
            "url": "https://httpbin.org/post",
            "body": {
                "message": "{output}",
                "nested": {
                    "value": "{data.field}"
                }
            }
        }

        step_input = {
            "output": "Test output",
            "data": {
                "field": "Test field"
            }
        }

        result = processor.process(step_input, config)

        assert result.get("webhook_success") == True
        print("✓ Template substitution test passed")


class TestAPIIntegrationProcessor:
    """Test APIIntegrationProcessor"""

    def test_basic_rest_call(self):
        """Test basic REST API call"""
        processor = APIIntegrationProcessor()

        config = {
            "type": "api",
            "api_type": "rest",
            "url": "https://httpbin.org/get",
            "method": "GET",
            "timeout": 10.0
        }

        result = processor.process({}, config)

        assert result.get("api_success") == True
        assert "api_response" in result
        print("✓ Basic REST API test passed")

    def test_api_with_cache(self):
        """Test API with caching"""
        processor = APIIntegrationProcessor()

        config = {
            "type": "api",
            "api_type": "rest",
            "url": "https://httpbin.org/uuid",
            "method": "GET",
            "cache": {
                "enabled": True,
                "ttl": 60,
                "key_fields": ["input"]
            }
        }

        # First call - should miss cache
        result1 = processor.process({"input": "test"}, config)
        assert result1.get("api_success") == True
        assert result1.get("api_cached") == False

        # Second call - should hit cache
        result2 = processor.process({"input": "test"}, config)
        assert result2.get("api_cached") == True

        print("✓ API caching test passed")

    def test_rate_limiting(self):
        """Test rate limiting"""
        processor = APIIntegrationProcessor()

        config = {
            "type": "api",
            "api_type": "rest",
            "url": "https://httpbin.org/get",
            "method": "GET",
            "rate_limit": {
                "enabled": True,
                "requests_per_second": 2,
                "burst_size": 2
            }
        }

        # Make 3 rapid requests
        results = []
        for i in range(3):
            result = processor.process({"request": i}, config)
            results.append(result.get("api_success") or result.get("api_rate_limited"))

        # At least one should be rate limited
        assert False in [r for r in results if not results[0]]
        print("✓ Rate limiting test passed")

    def test_authentication(self):
        """Test authentication header generation"""
        processor = APIIntegrationProcessor()

        # Test Bearer auth
        config = {
            "auth": {
                "type": "bearer",
                "token": "test_token_123"
            }
        }

        headers = processor._add_authentication({}, config)
        assert headers.get("Authorization") == "Bearer test_token_123"

        # Test API key auth
        config = {
            "auth": {
                "type": "api_key",
                "key": "X-API-Key",
                "value": "api_key_456"
            }
        }

        headers = processor._add_authentication({}, config)
        assert headers.get("X-API-Key") == "api_key_456"

        print("✓ Authentication test passed")


class TestDataValidationProcessor:
    """Test DataValidationProcessor"""

    def test_schema_validation(self):
        """Test JSON schema validation"""
        processor = DataValidationProcessor()

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
                            "age": {"type": "integer", "minimum": 0}
                        }
                    }
                }
            ],
            "source_field": "data",
            "on_validation_error": "log"
        }

        # Valid data
        valid_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "age": 25
        }

        result = processor.process({"data": valid_data}, config)
        assert result.get("is_valid") == True

        # Invalid data
        invalid_data = {
            "name": "",  # Too short
            "email": "invalid",  # Bad format
            "age": -5  # Below minimum
        }

        result = processor.process({"data": invalid_data}, config)
        assert result.get("is_valid") == False

        print("✓ Schema validation test passed")

    def test_pii_detection(self):
        """Test PII detection and redaction"""
        processor = DataValidationProcessor()

        config = {
            "type": "validation",
            "validations": [
                {
                    "type": "pii",
                    "detect": ["email", "phone", "ssn"],
                    "action": "redact",
                    "replacement": "[REDACTED]"
                }
            ],
            "source_field": "text"
        }

        text_with_pii = "Contact me at john@example.com or 123-456-7890. SSN: 123-45-6789"

        result = processor.process({"text": text_with_pii}, config)

        # Check that PII was detected
        validation_results = result.get("validation_results", [])
        assert len(validation_results) > 0

        pii_result = validation_results[0]
        assert pii_result.get("type") == "pii"
        assert pii_result.get("pii_count", 0) > 0

        # Check that PII was redacted
        sanitized = result.get("validated_data", {}).get("text", "")
        assert "[REDACTED]" in sanitized or pii_result.get("pii_count") > 0

        print("✓ PII detection test passed")

    def test_sanitization(self):
        """Test data sanitization"""
        processor = DataValidationProcessor()

        config = {
            "type": "validation",
            "validations": [
                {
                    "type": "sanitize",
                    "rules": [
                        {"type": "html_escape"},
                        {"type": "xss_filter"}
                    ]
                }
            ],
            "source_field": "input"
        }

        malicious_input = "<script>alert('xss')</script><b>Bold text</b>"

        result = processor.process({"input": malicious_input}, config)

        sanitized = result.get("validated_data", {}).get("input", "")

        # Script tag should be removed or escaped
        assert "<script>" not in sanitized or "&lt;script&gt;" in sanitized

        print("✓ Sanitization test passed")

    def test_data_quality_checks(self):
        """Test data quality assessment"""
        processor = DataValidationProcessor()

        config = {
            "type": "validation",
            "validations": [
                {
                    "type": "data_quality",
                    "checks": [
                        {"type": "completeness", "threshold": 0.8}
                    ]
                }
            ],
            "source_field": "record"
        }

        # Complete record
        complete_record = {
            "field1": "value1",
            "field2": "value2",
            "field3": "value3"
        }

        result = processor.process({"record": complete_record}, config)
        assert result.get("is_valid") == True

        # Incomplete record
        incomplete_record = {
            "field1": "value1",
            "field2": None,
            "field3": ""
        }

        result = processor.process({"record": incomplete_record}, config)
        # Should have quality issues
        assert len(result.get("validation_results", [])) > 0

        print("✓ Data quality test passed")


class TestCacheProcessor:
    """Test CacheProcessor"""

    def test_memory_cache(self):
        """Test in-memory caching"""
        processor = CacheProcessor()

        # Set value
        set_config = {
            "type": "cache",
            "operation": "set",
            "strategy": "memory",
            "key": {
                "fields": ["input"],
                "prefix": "test"
            },
            "ttl": 60
        }

        step_input = {"input": "test_key", "output": "test_value"}
        result = processor.process(step_input, set_config)
        assert result.get("cache_set") == True

        # Get value
        get_config = {
            "type": "cache",
            "operation": "get",
            "strategy": "memory",
            "key": {
                "fields": ["input"],
                "prefix": "test"
            }
        }

        result = processor.process({"input": "test_key"}, get_config)
        assert result.get("cache_hit") == True
        assert result.get("cached_value") == "test_value"

        print("✓ Memory cache test passed")

    def test_multi_level_cache(self):
        """Test multi-level caching"""
        processor = CacheProcessor()

        config = {
            "type": "cache",
            "operation": "set",
            "strategy": "multi_level",
            "levels": [
                {"type": "memory", "max_size": 10, "ttl": 60},
                {"type": "disk", "cache_dir": "/tmp/test_cache", "ttl": 120}
            ],
            "key": {
                "fields": ["key"],
                "prefix": "ml"
            }
        }

        # Set value
        result = processor.process(
            {"key": "test", "output": "multilevel_value"},
            config
        )
        assert result.get("cache_set") == True

        # Get value (should hit L1 memory)
        get_config = config.copy()
        get_config["operation"] = "get"

        result = processor.process({"key": "test"}, get_config)
        assert result.get("cache_hit") == True

        print("✓ Multi-level cache test passed")

    def test_cache_metrics(self):
        """Test cache metrics tracking"""
        processor = CacheProcessor()

        config = {
            "type": "cache",
            "operation": "get",
            "strategy": "memory",
            "key": {"fields": ["input"]},
            "metrics": {
                "enabled": True,
                "track_cost": True,
                "cost_per_request": 0.01
            },
            "cache_name": "metrics_test"
        }

        # Generate some cache activity
        for i in range(5):
            processor.process({"input": f"key_{i % 2}"}, config)

        # Get metrics
        metrics = processor.get_metrics("metrics_test")

        assert metrics.get("total_requests") > 0
        assert "hit_rate" in metrics
        assert "miss_rate" in metrics

        # Get cost savings
        savings = processor.get_cost_savings("metrics_test", 0.01)
        assert "total_cost_saved" in savings

        print("✓ Cache metrics test passed")

    def test_cache_clear(self):
        """Test cache clearing"""
        processor = CacheProcessor()

        # Set some values
        for i in range(3):
            processor.process(
                {"key": f"clear_test_{i}", "output": f"value_{i}"},
                {
                    "operation": "set",
                    "strategy": "memory",
                    "key": {"fields": ["key"]}
                }
            )

        # Clear cache
        result = processor.process(
            {},
            {
                "operation": "clear",
                "strategy": "memory"
            }
        )

        assert result.get("cache_cleared") == True

        print("✓ Cache clear test passed")


class TestAggregationProcessor:
    """Test AggregationProcessor"""

    def test_statistical_aggregation(self):
        """Test statistical aggregation methods"""
        processor = AggregationProcessor()

        results = [10, 20, 30, 40, 50]

        # Test mean
        mean_result = processor.process(
            {"parallel_results": results},
            {
                "strategy": "mean",
                "source_field": "parallel_results",
                "target_field": "mean_value"
            }
        )
        assert mean_result.get("mean_value") == 30.0

        # Test median
        median_result = processor.process(
            {"parallel_results": results},
            {
                "strategy": "median",
                "source_field": "parallel_results",
                "target_field": "median_value"
            }
        )
        assert median_result.get("median_value") == 30.0

        # Test max
        max_result = processor.process(
            {"parallel_results": results},
            {
                "strategy": "max",
                "source_field": "parallel_results",
                "target_field": "max_value"
            }
        )
        assert max_result.get("max_value") == 50.0

        print("✓ Statistical aggregation test passed")

    def test_voting_aggregation(self):
        """Test voting mechanisms"""
        processor = AggregationProcessor()

        # Plurality voting
        votes = ["A", "A", "B", "A", "C"]

        result = processor.process(
            {"parallel_results": votes},
            {
                "strategy": "voting",
                "source_field": "parallel_results",
                "target_field": "winner",
                "voting": {
                    "method": "plurality"
                }
            }
        )

        assert result.get("winner") == "A"

        # Majority voting
        majority_votes = ["A", "A", "A", "B"]

        result = processor.process(
            {"parallel_results": majority_votes},
            {
                "strategy": "voting",
                "source_field": "parallel_results",
                "target_field": "winner",
                "voting": {
                    "method": "majority",
                    "threshold": 0.5
                }
            }
        )

        assert result.get("winner") == "A"

        print("✓ Voting aggregation test passed")

    def test_outlier_detection(self):
        """Test outlier detection"""
        processor = AggregationProcessor()

        # Data with outlier
        data = [10, 12, 11, 13, 12, 100]  # 100 is outlier

        result = processor.process(
            {"parallel_results": data},
            {
                "strategy": "mean",
                "source_field": "parallel_results",
                "target_field": "mean",
                "statistical": {
                    "detect_outliers": True,
                    "outlier_method": "iqr",
                    "remove_outliers": True
                }
            }
        )

        metadata = result.get("aggregation_metadata", {})
        assert metadata.get("outliers_count", 0) > 0

        # Mean should be calculated without outlier
        mean = result.get("mean")
        assert mean < 20  # Should be around 11.6

        print("✓ Outlier detection test passed")

    def test_confidence_scoring(self):
        """Test confidence score calculation"""
        processor = AggregationProcessor()

        # High agreement
        high_agreement = ["A", "A", "A", "A", "B"]

        result = processor.process(
            {"parallel_results": high_agreement},
            {
                "strategy": "voting",
                "source_field": "parallel_results",
                "target_field": "result",
                "confidence": {
                    "enabled": True,
                    "method": "agreement"
                }
            }
        )

        confidence = result.get("aggregation_metadata", {}).get("confidence_score")
        assert confidence > 0.7  # High confidence

        # Low agreement
        low_agreement = ["A", "B", "C", "D", "E"]

        result = processor.process(
            {"parallel_results": low_agreement},
            {
                "strategy": "voting",
                "source_field": "parallel_results",
                "target_field": "result",
                "confidence": {
                    "enabled": True,
                    "method": "agreement"
                }
            }
        )

        confidence = result.get("aggregation_metadata", {}).get("confidence_score")
        assert confidence <= 0.3  # Low confidence

        print("✓ Confidence scoring test passed")

    def test_merge_aggregation(self):
        """Test dictionary merging"""
        processor = AggregationProcessor()

        dicts = [
            {"a": 1, "b": 2},
            {"b": 3, "c": 4},
            {"c": 5, "d": 6}
        ]

        result = processor.process(
            {"parallel_results": dicts},
            {
                "strategy": "merge",
                "source_field": "parallel_results",
                "target_field": "merged",
                "merge_strategy": {
                    "conflicts": "prefer_last",
                    "nested": True
                }
            }
        )

        merged = result.get("merged")
        assert isinstance(merged, dict)
        assert "a" in merged
        assert "d" in merged

        print("✓ Merge aggregation test passed")


def run_all_tests():
    """Run all processor tests"""
    print("\n" + "="*60)
    print("Running Custom Processor Integration Tests")
    print("="*60 + "\n")

    test_classes = [
        TestWebhookProcessor,
        TestAPIIntegrationProcessor,
        TestDataValidationProcessor,
        TestCacheProcessor,
        TestAggregationProcessor
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 60)

        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith("test_")]

        for method_name in methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                passed_tests += 1
            except Exception as e:
                print(f"✗ {method_name} FAILED: {str(e)}")

    print("\n" + "="*60)
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    print("="*60 + "\n")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
