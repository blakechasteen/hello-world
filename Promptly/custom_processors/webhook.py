#!/usr/bin/env python3
"""
Webhook Processor for Promptly Chains

Send HTTP webhooks during chain execution with retry logic,
signature verification, and response validation.
"""

import time
import json
import hmac
import hashlib
import requests
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from promptly.plugins.base import BaseChainStepProcessor


class HTTPMethod(Enum):
    """Supported HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class WebhookProcessor(BaseChainStepProcessor):
    """
    Send HTTP webhooks during chain execution.

    Configuration format:
    {
        "type": "webhook",
        "url": "https://api.example.com/webhook",
        "method": "POST",  # GET, POST, PUT, PATCH, DELETE
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer token"
        },
        "body": {
            "event": "step_completed",
            "data": "{output}"  # Template variables
        },
        "query_params": {
            "api_key": "xxx"
        },
        "timeout": 30.0,  # seconds
        "retry": {
            "enabled": true,
            "max_attempts": 3,
            "backoff_strategy": "exponential",
            "initial_delay": 1.0,
            "max_delay": 30.0,
            "multiplier": 2.0
        },
        "signature": {
            "enabled": true,
            "secret": "webhook_secret",
            "algorithm": "sha256",  # sha256, sha1, md5
            "header": "X-Webhook-Signature"
        },
        "response_validation": {
            "enabled": true,
            "expected_status": [200, 201, 202],
            "validate_json": true,
            "json_schema": {...}
        },
        "on_success": "continue",  # continue, store, callback
        "on_failure": "log"  # log, throw, fallback
    }

    Examples:
    - Slack notifications
    - Discord webhooks
    - Custom API callbacks
    - Event logging
    """

    def __init__(self):
        super().__init__(
            name="webhook",
            description="Send HTTP webhooks with retry logic and signature verification"
        )
        self._callbacks: Dict[str, Callable] = {}

    def register_callback(self, name: str, callback: Callable) -> None:
        """Register a callback function for webhook events"""
        self._callbacks[name] = callback

    def process(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send webhook and handle response.

        Args:
            step_input: Input data from previous step
            step_config: Webhook configuration

        Returns:
            Step input with webhook result added
        """
        url = step_config.get("url")
        if not url:
            return {
                **step_input,
                "webhook_error": "No webhook URL provided",
                "webhook_success": False
            }

        # Build request
        method = step_config.get("method", "POST").upper()
        headers = self._build_headers(step_config, step_input)
        body = self._build_body(step_config, step_input)
        query_params = self._build_query_params(step_config, step_input)
        timeout = step_config.get("timeout", 30.0)

        # Retry configuration
        retry_config = step_config.get("retry", {})
        if retry_config.get("enabled", True):
            result = self._send_with_retry(
                url=url,
                method=method,
                headers=headers,
                body=body,
                params=query_params,
                timeout=timeout,
                retry_config=retry_config,
                step_config=step_config
            )
        else:
            result = self._send_request(
                url=url,
                method=method,
                headers=headers,
                body=body,
                params=query_params,
                timeout=timeout,
                step_config=step_config
            )

        # Handle result
        on_success = step_config.get("on_success", "continue")
        on_failure = step_config.get("on_failure", "log")

        if result.get("success"):
            return self._handle_success(step_input, result, on_success)
        else:
            return self._handle_failure(step_input, result, on_failure)

    def _build_headers(self, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, str]:
        """Build HTTP headers with template substitution"""
        headers = config.get("headers", {}).copy()

        # Template substitution
        for key, value in headers.items():
            if isinstance(value, str):
                headers[key] = self._substitute_template(value, data)

        # Add default headers
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        headers["User-Agent"] = "Promptly-Webhook/1.0"

        return headers

    def _build_body(self, config: Dict[str, Any], data: Dict[str, Any]) -> Optional[str]:
        """Build request body with template substitution"""
        body = config.get("body")

        if not body:
            return None

        # If body is a dict, perform template substitution and convert to JSON
        if isinstance(body, dict):
            body = self._substitute_dict_templates(body, data)
            return json.dumps(body)

        # If body is a string, perform substitution
        if isinstance(body, str):
            return self._substitute_template(body, data)

        return str(body)

    def _build_query_params(self, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, str]:
        """Build query parameters with template substitution"""
        params = config.get("query_params", {}).copy()

        for key, value in params.items():
            if isinstance(value, str):
                params[key] = self._substitute_template(value, data)

        return params

    def _substitute_template(self, template: str, data: Dict[str, Any]) -> str:
        """Substitute template variables like {output}, {result.field}"""
        import re

        def replacer(match):
            field_path = match.group(1)
            value = self._get_field(data, field_path)
            return str(value) if value is not None else ""

        return re.sub(r'\{([^}]+)\}', replacer, template)

    def _substitute_dict_templates(self, template_dict: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively substitute templates in a dictionary"""
        result = {}
        for key, value in template_dict.items():
            if isinstance(value, str):
                result[key] = self._substitute_template(value, data)
            elif isinstance(value, dict):
                result[key] = self._substitute_dict_templates(value, data)
            elif isinstance(value, list):
                result[key] = [
                    self._substitute_template(item, data) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def _get_field(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get nested field value using dot notation"""
        parts = field_path.split(".")
        value = data

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None

        return value

    def _send_request(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        body: Optional[str],
        params: Dict[str, str],
        timeout: float,
        step_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send HTTP request"""
        try:
            # Add signature if configured
            signature_config = step_config.get("signature", {})
            if signature_config.get("enabled", False):
                headers = self._add_signature(headers, body, signature_config)

            # Send request
            start_time = time.time()

            if method == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=timeout)
            elif method == "POST":
                response = requests.post(url, headers=headers, data=body, params=params, timeout=timeout)
            elif method == "PUT":
                response = requests.put(url, headers=headers, data=body, params=params, timeout=timeout)
            elif method == "PATCH":
                response = requests.patch(url, headers=headers, data=body, params=params, timeout=timeout)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers, params=params, timeout=timeout)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported HTTP method: {method}"
                }

            duration = time.time() - start_time

            # Validate response
            validation_config = step_config.get("response_validation", {})
            validation_result = self._validate_response(response, validation_config)

            return {
                "success": validation_result.get("valid", True),
                "status_code": response.status_code,
                "response_body": response.text,
                "response_json": self._try_parse_json(response.text),
                "duration": duration,
                "validation": validation_result,
                "timestamp": datetime.now().isoformat()
            }

        except requests.Timeout:
            return {
                "success": False,
                "error": "Request timeout",
                "error_type": "timeout"
            }
        except requests.ConnectionError as e:
            return {
                "success": False,
                "error": f"Connection error: {str(e)}",
                "error_type": "connection"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": "unknown"
            }

    def _send_with_retry(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        body: Optional[str],
        params: Dict[str, str],
        timeout: float,
        retry_config: Dict[str, Any],
        step_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send request with exponential backoff retry"""
        max_attempts = retry_config.get("max_attempts", 3)
        backoff_strategy = retry_config.get("backoff_strategy", "exponential")
        initial_delay = retry_config.get("initial_delay", 1.0)
        max_delay = retry_config.get("max_delay", 30.0)
        multiplier = retry_config.get("multiplier", 2.0)

        attempts = 0
        retry_history = []

        while attempts < max_attempts:
            attempts += 1

            # Calculate delay for this attempt
            if attempts > 1:
                delay = self._calculate_delay(
                    attempt=attempts - 1,
                    strategy=backoff_strategy,
                    initial_delay=initial_delay,
                    max_delay=max_delay,
                    multiplier=multiplier
                )
                time.sleep(delay)
                retry_history.append({"attempt": attempts, "delay": delay})

            # Send request
            result = self._send_request(
                url=url,
                method=method,
                headers=headers,
                body=body,
                params=params,
                timeout=timeout,
                step_config=step_config
            )

            # Check if successful
            if result.get("success"):
                result["attempts"] = attempts
                result["retry_history"] = retry_history
                return result

            # Record failure
            retry_history.append({
                "attempt": attempts,
                "error": result.get("error"),
                "status_code": result.get("status_code")
            })

            # Check if should retry
            if not self._should_retry(result, retry_config):
                break

        # All attempts exhausted
        return {
            "success": False,
            "error": "Max retry attempts exhausted",
            "attempts": attempts,
            "retry_history": retry_history
        }

    def _calculate_delay(
        self,
        attempt: int,
        strategy: str,
        initial_delay: float,
        max_delay: float,
        multiplier: float
    ) -> float:
        """Calculate retry delay based on strategy"""
        import random

        if strategy == "constant":
            delay = initial_delay
        elif strategy == "linear":
            delay = initial_delay * (attempt + 1)
        elif strategy == "exponential":
            delay = initial_delay * (multiplier ** attempt)
        elif strategy == "fibonacci":
            fib = [1, 1]
            for i in range(2, attempt + 2):
                fib.append(fib[i-1] + fib[i-2])
            delay = initial_delay * fib[min(attempt + 1, len(fib) - 1)]
        elif strategy == "jitter":
            base_delay = initial_delay * (multiplier ** attempt)
            jitter = random.uniform(-0.1, 0.1) * base_delay
            delay = base_delay + jitter
        else:
            delay = initial_delay

        return min(delay, max_delay)

    def _should_retry(self, result: Dict[str, Any], retry_config: Dict[str, Any]) -> bool:
        """Check if request should be retried"""
        # Retry on specific error types
        retry_on_errors = retry_config.get("retry_on_errors", ["timeout", "connection"])
        error_type = result.get("error_type")

        if error_type in retry_on_errors:
            return True

        # Retry on specific status codes
        retry_on_status = retry_config.get("retry_on_status", [429, 500, 502, 503, 504])
        status_code = result.get("status_code")

        if status_code in retry_on_status:
            return True

        return False

    def _add_signature(
        self,
        headers: Dict[str, str],
        body: Optional[str],
        signature_config: Dict[str, Any]
    ) -> Dict[str, str]:
        """Add HMAC signature to headers"""
        secret = signature_config.get("secret", "")
        algorithm = signature_config.get("algorithm", "sha256")
        header_name = signature_config.get("header", "X-Webhook-Signature")

        if not secret:
            return headers

        # Generate signature
        payload = body.encode('utf-8') if body else b''

        if algorithm == "sha256":
            signature = hmac.new(secret.encode('utf-8'), payload, hashlib.sha256).hexdigest()
        elif algorithm == "sha1":
            signature = hmac.new(secret.encode('utf-8'), payload, hashlib.sha1).hexdigest()
        elif algorithm == "md5":
            signature = hmac.new(secret.encode('utf-8'), payload, hashlib.md5).hexdigest()
        else:
            signature = hmac.new(secret.encode('utf-8'), payload, hashlib.sha256).hexdigest()

        # Add to headers
        headers = headers.copy()
        headers[header_name] = f"{algorithm}={signature}"

        return headers

    def verify_signature(
        self,
        body: str,
        signature: str,
        secret: str,
        algorithm: str = "sha256"
    ) -> bool:
        """Verify HMAC signature (for webhook receivers)"""
        # Extract algorithm and signature
        if "=" in signature:
            alg, sig = signature.split("=", 1)
        else:
            alg = algorithm
            sig = signature

        # Generate expected signature
        payload = body.encode('utf-8')

        if alg == "sha256":
            expected = hmac.new(secret.encode('utf-8'), payload, hashlib.sha256).hexdigest()
        elif alg == "sha1":
            expected = hmac.new(secret.encode('utf-8'), payload, hashlib.sha1).hexdigest()
        elif alg == "md5":
            expected = hmac.new(secret.encode('utf-8'), payload, hashlib.md5).hexdigest()
        else:
            return False

        # Constant-time comparison
        return hmac.compare_digest(expected, sig)

    def _validate_response(
        self,
        response: requests.Response,
        validation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate webhook response"""
        if not validation_config.get("enabled", False):
            return {"valid": True}

        errors = []

        # Check status code
        expected_status = validation_config.get("expected_status", [200, 201, 202])
        if response.status_code not in expected_status:
            errors.append(f"Unexpected status code: {response.status_code}")

        # Validate JSON
        if validation_config.get("validate_json", False):
            try:
                json_data = response.json()

                # Check JSON schema if provided
                json_schema = validation_config.get("json_schema")
                if json_schema:
                    schema_errors = self._validate_json_schema(json_data, json_schema)
                    errors.extend(schema_errors)

            except ValueError:
                errors.append("Response is not valid JSON")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    def _validate_json_schema(self, data: Any, schema: Dict[str, Any]) -> List[str]:
        """Validate JSON against simple schema"""
        errors = []

        # Simple schema validation (required fields, types)
        if "required" in schema:
            for field in schema["required"]:
                if field not in data:
                    errors.append(f"Required field missing: {field}")

        if "properties" in schema:
            for field, props in schema["properties"].items():
                if field in data:
                    expected_type = props.get("type")
                    if expected_type:
                        if not self._check_type(data[field], expected_type):
                            errors.append(f"Field '{field}' has wrong type")

        return errors

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict
        }

        expected = type_map.get(expected_type)
        if expected:
            return isinstance(value, expected)

        return True

    def _try_parse_json(self, text: str) -> Optional[Any]:
        """Try to parse JSON, return None if invalid"""
        try:
            return json.loads(text)
        except:
            return None

    def _handle_success(
        self,
        step_input: Dict[str, Any],
        result: Dict[str, Any],
        on_success: str
    ) -> Dict[str, Any]:
        """Handle successful webhook"""
        output = {
            **step_input,
            "webhook_success": True,
            "webhook_result": result
        }

        if on_success == "store":
            # Store full result
            output["webhook_response"] = result.get("response_json") or result.get("response_body")

        elif on_success == "callback":
            # Execute callback
            callback_name = result.get("callback")
            if callback_name and callback_name in self._callbacks:
                self._callbacks[callback_name](result)

        # Default: continue
        return output

    def _handle_failure(
        self,
        step_input: Dict[str, Any],
        result: Dict[str, Any],
        on_failure: str
    ) -> Dict[str, Any]:
        """Handle failed webhook"""
        if on_failure == "throw":
            raise Exception(f"Webhook failed: {result.get('error')}")

        elif on_failure == "fallback":
            # Return fallback data
            return {
                **step_input,
                "webhook_success": False,
                "webhook_error": result.get("error"),
                "webhook_used_fallback": True
            }

        # Default: log
        return {
            **step_input,
            "webhook_success": False,
            "webhook_error": result.get("error"),
            "webhook_result": result
        }


# Convenience functions for common webhooks

def create_slack_webhook(webhook_url: str, channel: Optional[str] = None) -> Dict[str, Any]:
    """Create configuration for Slack webhook"""
    config = {
        "type": "webhook",
        "url": webhook_url,
        "method": "POST",
        "headers": {
            "Content-Type": "application/json"
        },
        "body": {
            "text": "{output}"
        }
    }

    if channel:
        config["body"]["channel"] = channel

    return config


def create_discord_webhook(webhook_url: str) -> Dict[str, Any]:
    """Create configuration for Discord webhook"""
    return {
        "type": "webhook",
        "url": webhook_url,
        "method": "POST",
        "headers": {
            "Content-Type": "application/json"
        },
        "body": {
            "content": "{output}"
        }
    }


__all__ = [
    "WebhookProcessor",
    "HTTPMethod",
    "create_slack_webhook",
    "create_discord_webhook"
]
