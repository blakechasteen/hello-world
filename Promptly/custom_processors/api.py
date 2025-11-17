#!/usr/bin/env python3
"""
API Integration Processor for Promptly Chains

Call external APIs (REST, GraphQL, gRPC) with authentication,
rate limiting, response caching, and error handling.
"""

import time
import json
import hashlib
import requests
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from promptly.plugins.base import BaseChainStepProcessor


class APIType(Enum):
    """Supported API types"""
    REST = "rest"
    GRAPHQL = "graphql"
    GRPC = "grpc"
    SOAP = "soap"


class AuthType(Enum):
    """Authentication types"""
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    BASIC = "basic"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    CUSTOM = "custom"


class RateLimiter:
    """Token bucket rate limiter with burst support"""

    def __init__(self, requests_per_second: float, burst_size: int):
        self.rate = requests_per_second
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_update = time.time()

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens, return True if successful"""
        now = time.time()
        elapsed = now - self.last_update

        # Add tokens based on elapsed time
        self.tokens = min(
            self.burst_size,
            self.tokens + elapsed * self.rate
        )
        self.last_update = now

        # Check if enough tokens available
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False

    def wait_time(self) -> float:
        """Calculate wait time for next request"""
        if self.tokens >= 1.0:
            return 0.0

        tokens_needed = 1.0 - self.tokens
        return tokens_needed / self.rate


class ResponseCache:
    """Simple in-memory cache with TTL"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: deque = deque()

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key not in self.cache:
            return None

        entry = self.cache[key]
        expires_at = entry.get("expires_at")

        if expires_at and datetime.now() > expires_at:
            # Expired
            del self.cache[key]
            return None

        # Update access time
        entry["last_accessed"] = datetime.now()
        return entry.get("value")

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with TTL"""
        # Evict if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        ttl = ttl if ttl is not None else self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None

        self.cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": datetime.now(),
            "last_accessed": datetime.now()
        }

    def invalidate(self, key: str) -> None:
        """Invalidate cache entry"""
        if key in self.cache:
            del self.cache[key]

    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()

    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if not self.cache:
            return

        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k]["last_accessed"]
        )
        del self.cache[lru_key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "entries": list(self.cache.keys())
        }


class APIIntegrationProcessor(BaseChainStepProcessor):
    """
    Call external APIs with authentication and rate limiting.

    Configuration format:
    {
        "type": "api",
        "api_type": "rest",  # rest, graphql, grpc, soap
        "url": "https://api.example.com/v1/endpoint",
        "method": "POST",
        "auth": {
            "type": "bearer",  # none, api_key, bearer, basic, oauth2, jwt
            "token": "your_token",
            "header": "Authorization",
            "prefix": "Bearer",
            # For API key
            "key": "api_key",
            "value": "xxx",
            # For Basic
            "username": "user",
            "password": "pass",
            # For OAuth2
            "access_token": "token",
            "refresh_token": "refresh",
            "token_url": "https://oauth.example.com/token"
        },
        "headers": {
            "Content-Type": "application/json"
        },
        "body": {
            "prompt": "{input}",
            "max_tokens": 100
        },
        "query_params": {},
        "timeout": 30.0,
        "rate_limit": {
            "enabled": true,
            "requests_per_second": 10,
            "burst_size": 20
        },
        "cache": {
            "enabled": true,
            "ttl": 300,  # seconds
            "key_fields": ["input"],  # fields to use for cache key
            "strategy": "memory"  # memory, redis, disk
        },
        "retry": {
            "enabled": true,
            "max_attempts": 3,
            "backoff_strategy": "exponential"
        },
        "fallback": {
            "enabled": true,
            "value": "default response"
        },
        "response_mapping": {
            "output": "choices[0].message.content",
            "usage": "usage.total_tokens"
        }
    }

    Examples for popular APIs:
    - OpenAI
    - Anthropic
    - Google AI
    - Custom REST APIs
    """

    def __init__(self):
        super().__init__(
            name="api",
            description="Call external APIs with authentication, rate limiting, and caching"
        )
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._caches: Dict[str, ResponseCache] = {}
        self._auth_providers: Dict[str, Callable] = {}

    def register_auth_provider(self, name: str, provider: Callable) -> None:
        """Register custom authentication provider"""
        self._auth_providers[name] = provider

    def process(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute API call with authentication, rate limiting, and caching.

        Args:
            step_input: Input data from previous step
            step_config: API configuration

        Returns:
            Step input with API response added
        """
        api_type = step_config.get("api_type", "rest")

        if api_type == APIType.REST.value:
            return self._call_rest_api(step_input, step_config)
        elif api_type == APIType.GRAPHQL.value:
            return self._call_graphql_api(step_input, step_config)
        elif api_type == APIType.GRPC.value:
            return self._call_grpc_api(step_input, step_config)
        else:
            return {
                **step_input,
                "api_error": f"Unsupported API type: {api_type}",
                "api_success": False
            }

    def _call_rest_api(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Call REST API"""
        url = step_config.get("url")
        if not url:
            return {
                **step_input,
                "api_error": "No API URL provided",
                "api_success": False
            }

        # Check cache
        cache_config = step_config.get("cache", {})
        if cache_config.get("enabled", False):
            cache_key = self._generate_cache_key(step_input, cache_config)
            cached_response = self._get_cached_response(cache_key, cache_config)

            if cached_response is not None:
                return {
                    **step_input,
                    "api_response": cached_response,
                    "api_success": True,
                    "api_cached": True
                }

        # Rate limiting
        rate_config = step_config.get("rate_limit", {})
        if rate_config.get("enabled", False):
            if not self._check_rate_limit(rate_config):
                return {
                    **step_input,
                    "api_error": "Rate limit exceeded",
                    "api_success": False,
                    "api_rate_limited": True
                }

        # Build request
        method = step_config.get("method", "GET").upper()
        headers = self._build_headers(step_config, step_input)
        body = self._build_body(step_config, step_input)
        params = self._build_params(step_config, step_input)
        timeout = step_config.get("timeout", 30.0)

        # Add authentication
        headers = self._add_authentication(headers, step_config)

        # Make request (with retry if configured)
        retry_config = step_config.get("retry", {})
        if retry_config.get("enabled", False):
            result = self._request_with_retry(
                url, method, headers, body, params, timeout, retry_config
            )
        else:
            result = self._make_request(
                url, method, headers, body, params, timeout
            )

        # Handle result
        if result.get("success"):
            response = result.get("response_json") or result.get("response_body")

            # Map response fields
            mapped_response = self._map_response(response, step_config)

            # Cache response
            if cache_config.get("enabled", False):
                self._cache_response(cache_key, mapped_response, cache_config)

            return {
                **step_input,
                "api_response": mapped_response,
                "api_raw_response": response,
                "api_success": True,
                "api_cached": False,
                "api_metadata": {
                    "status_code": result.get("status_code"),
                    "duration": result.get("duration")
                }
            }
        else:
            # Handle failure with fallback
            fallback_config = step_config.get("fallback", {})
            if fallback_config.get("enabled", False):
                return {
                    **step_input,
                    "api_response": fallback_config.get("value"),
                    "api_success": True,
                    "api_fallback_used": True,
                    "api_error": result.get("error")
                }
            else:
                return {
                    **step_input,
                    "api_error": result.get("error"),
                    "api_success": False
                }

    def _call_graphql_api(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Call GraphQL API"""
        url = step_config.get("url")
        query = step_config.get("query", "")
        variables = step_config.get("variables", {})

        # Substitute variables
        variables = self._substitute_dict_templates(variables, step_input)

        # Build GraphQL request
        graphql_config = step_config.copy()
        graphql_config["method"] = "POST"
        graphql_config["body"] = {
            "query": query,
            "variables": variables
        }

        # Use REST API call with GraphQL payload
        result = self._call_rest_api(step_input, graphql_config)

        # Extract data from GraphQL response
        if result.get("api_success") and "api_response" in result:
            response = result["api_response"]
            if isinstance(response, dict) and "data" in response:
                result["api_response"] = response["data"]
                if "errors" in response:
                    result["api_errors"] = response["errors"]

        return result

    def _call_grpc_api(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Call gRPC API (basic implementation)"""
        return {
            **step_input,
            "api_error": "gRPC support not yet implemented",
            "api_success": False
        }

    def _make_request(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        body: Optional[str],
        params: Dict[str, str],
        timeout: float
    ) -> Dict[str, Any]:
        """Make HTTP request"""
        try:
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
                return {"success": False, "error": f"Unsupported method: {method}"}

            duration = time.time() - start_time

            # Check status
            if response.status_code >= 400:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "status_code": response.status_code
                }

            return {
                "success": True,
                "status_code": response.status_code,
                "response_body": response.text,
                "response_json": self._try_parse_json(response.text),
                "duration": duration
            }

        except requests.Timeout:
            return {"success": False, "error": "Request timeout"}
        except requests.ConnectionError as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _request_with_retry(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        body: Optional[str],
        params: Dict[str, str],
        timeout: float,
        retry_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make request with retry logic"""
        max_attempts = retry_config.get("max_attempts", 3)
        backoff_strategy = retry_config.get("backoff_strategy", "exponential")
        initial_delay = retry_config.get("initial_delay", 1.0)

        for attempt in range(max_attempts):
            result = self._make_request(url, method, headers, body, params, timeout)

            if result.get("success"):
                return result

            # Wait before retry
            if attempt < max_attempts - 1:
                delay = self._calculate_backoff(attempt, backoff_strategy, initial_delay)
                time.sleep(delay)

        return result

    def _calculate_backoff(self, attempt: int, strategy: str, initial_delay: float) -> float:
        """Calculate backoff delay"""
        if strategy == "exponential":
            return initial_delay * (2 ** attempt)
        elif strategy == "linear":
            return initial_delay * (attempt + 1)
        else:
            return initial_delay

    def _build_headers(self, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, str]:
        """Build HTTP headers"""
        headers = config.get("headers", {}).copy()

        # Template substitution
        for key, value in headers.items():
            if isinstance(value, str):
                headers[key] = self._substitute_template(value, data)

        return headers

    def _build_body(self, config: Dict[str, Any], data: Dict[str, Any]) -> Optional[str]:
        """Build request body"""
        body = config.get("body")

        if not body:
            return None

        if isinstance(body, dict):
            body = self._substitute_dict_templates(body, data)
            return json.dumps(body)

        if isinstance(body, str):
            return self._substitute_template(body, data)

        return str(body)

    def _build_params(self, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, str]:
        """Build query parameters"""
        params = config.get("query_params", {}).copy()

        for key, value in params.items():
            if isinstance(value, str):
                params[key] = self._substitute_template(value, data)

        return params

    def _add_authentication(self, headers: Dict[str, str], config: Dict[str, Any]) -> Dict[str, str]:
        """Add authentication to headers"""
        auth_config = config.get("auth", {})
        auth_type = auth_config.get("type", "none")

        headers = headers.copy()

        if auth_type == AuthType.BEARER.value:
            token = auth_config.get("token", "")
            header = auth_config.get("header", "Authorization")
            prefix = auth_config.get("prefix", "Bearer")
            headers[header] = f"{prefix} {token}"

        elif auth_type == AuthType.API_KEY.value:
            key = auth_config.get("key", "X-API-Key")
            value = auth_config.get("value", "")
            headers[key] = value

        elif auth_type == AuthType.BASIC.value:
            import base64
            username = auth_config.get("username", "")
            password = auth_config.get("password", "")
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"

        elif auth_type == AuthType.JWT.value:
            token = auth_config.get("token", "")
            headers["Authorization"] = f"Bearer {token}"

        elif auth_type == AuthType.CUSTOM.value:
            # Use custom provider
            provider_name = auth_config.get("provider")
            if provider_name and provider_name in self._auth_providers:
                headers = self._auth_providers[provider_name](headers, auth_config)

        return headers

    def _check_rate_limit(self, rate_config: Dict[str, Any]) -> bool:
        """Check and apply rate limiting"""
        rate_key = rate_config.get("key", "default")

        if rate_key not in self._rate_limiters:
            self._rate_limiters[rate_key] = RateLimiter(
                requests_per_second=rate_config.get("requests_per_second", 10),
                burst_size=rate_config.get("burst_size", 20)
            )

        limiter = self._rate_limiters[rate_key]
        return limiter.acquire()

    def _generate_cache_key(self, data: Dict[str, Any], cache_config: Dict[str, Any]) -> str:
        """Generate cache key from input data"""
        key_fields = cache_config.get("key_fields", ["input"])

        # Extract field values
        values = []
        for field in key_fields:
            value = self._get_field(data, field)
            values.append(str(value))

        # Create hash
        key_string = "|".join(values)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str, cache_config: Dict[str, Any]) -> Optional[Any]:
        """Get response from cache"""
        cache_name = cache_config.get("name", "default")

        if cache_name not in self._caches:
            self._caches[cache_name] = ResponseCache(
                max_size=cache_config.get("max_size", 1000),
                default_ttl=cache_config.get("ttl", 300)
            )

        cache = self._caches[cache_name]
        return cache.get(cache_key)

    def _cache_response(self, cache_key: str, response: Any, cache_config: Dict[str, Any]) -> None:
        """Cache API response"""
        cache_name = cache_config.get("name", "default")

        if cache_name not in self._caches:
            self._caches[cache_name] = ResponseCache(
                max_size=cache_config.get("max_size", 1000),
                default_ttl=cache_config.get("ttl", 300)
            )

        cache = self._caches[cache_name]
        cache.set(cache_key, response, cache_config.get("ttl"))

    def _map_response(self, response: Any, config: Dict[str, Any]) -> Any:
        """Map response fields to output"""
        mapping = config.get("response_mapping", {})

        if not mapping:
            return response

        result = {}
        for output_field, input_path in mapping.items():
            value = self._get_field({"response": response}, f"response.{input_path}")
            result[output_field] = value

        return result

    def _substitute_template(self, template: str, data: Dict[str, Any]) -> str:
        """Substitute template variables"""
        import re

        def replacer(match):
            field_path = match.group(1)
            value = self._get_field(data, field_path)
            return str(value) if value is not None else ""

        return re.sub(r'\{([^}]+)\}', replacer, template)

    def _substitute_dict_templates(self, template_dict: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively substitute templates in dictionary"""
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
        """Get nested field value"""
        parts = field_path.split(".")
        value = data

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list):
                # Support array indexing like "items[0]"
                if "[" in part and "]" in part:
                    field_name = part[:part.index("[")]
                    index = int(part[part.index("[")+1:part.index("]")])
                    value = value.get(field_name, [])[index] if isinstance(value, dict) else value[index]
                else:
                    return None
            else:
                return None

        return value

    def _try_parse_json(self, text: str) -> Optional[Any]:
        """Try to parse JSON"""
        try:
            return json.loads(text)
        except:
            return None

    def get_cache_stats(self, cache_name: str = "default") -> Dict[str, Any]:
        """Get cache statistics"""
        if cache_name in self._caches:
            return self._caches[cache_name].get_stats()
        return {"error": "Cache not found"}

    def clear_cache(self, cache_name: str = "default") -> None:
        """Clear cache"""
        if cache_name in self._caches:
            self._caches[cache_name].clear()


# Convenience functions for popular APIs

def create_openai_config(api_key: str, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
    """Create configuration for OpenAI API"""
    return {
        "type": "api",
        "api_type": "rest",
        "url": "https://api.openai.com/v1/chat/completions",
        "method": "POST",
        "auth": {
            "type": "bearer",
            "token": api_key
        },
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": "{input}"}]
        },
        "response_mapping": {
            "output": "choices[0].message.content",
            "usage": "usage"
        },
        "cache": {
            "enabled": True,
            "ttl": 3600,
            "key_fields": ["input"]
        },
        "rate_limit": {
            "enabled": True,
            "requests_per_second": 3,
            "burst_size": 5
        }
    }


def create_anthropic_config(api_key: str, model: str = "claude-3-sonnet-20240229") -> Dict[str, Any]:
    """Create configuration for Anthropic API"""
    return {
        "type": "api",
        "api_type": "rest",
        "url": "https://api.anthropic.com/v1/messages",
        "method": "POST",
        "auth": {
            "type": "api_key",
            "key": "x-api-key",
            "value": api_key
        },
        "headers": {
            "anthropic-version": "2023-06-01"
        },
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": "{input}"}],
            "max_tokens": 1024
        },
        "response_mapping": {
            "output": "content[0].text",
            "usage": "usage"
        }
    }


__all__ = [
    "APIIntegrationProcessor",
    "APIType",
    "AuthType",
    "RateLimiter",
    "ResponseCache",
    "create_openai_config",
    "create_anthropic_config"
]
