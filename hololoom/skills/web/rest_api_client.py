"""REST API Client Skill - Generic HTTP operations"""

import asyncio
import time
from typing import Any

from hololoom.skills.base import (
    BaseSkill,
    SkillCategory,
    SkillInput,
    SkillMetadata,
    SkillOutput,
    SkillStatus,
)

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class RESTAPIClientSkill(BaseSkill):
    """Generic HTTP/REST API client."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="rest_api_client",
            version="1.0.0",
            author="HoloLoom Team",
            category=SkillCategory.WEB,
            tags=["http", "rest", "api", "requests", "web"],
            description_short="Generic HTTP/REST API client for web service integration",
            description_long="Make HTTP requests (GET/POST/PUT/PATCH/DELETE) to REST APIs with headers, authentication, JSON/form data, and response parsing.",
            requires_code_execution=True,
            requires_network=True,
            external_dependencies=["aiohttp or requests"],
            expected_latency_ms="100-5000ms",
            token_usage="100-1000 tokens"
        )

    def validate_input(self, skill_input: SkillInput) -> bool:
        valid_ops = ["get", "post", "put", "patch", "delete", "head"]
        if skill_input.operation not in valid_ops:
            raise ValueError(f"Invalid operation. Must be one of: {', '.join(valid_ops)}")
        if "url" not in skill_input.parameters:
            raise ValueError(f"{skill_input.operation} requires 'url' parameter")
        return True

    async def execute(self, skill_input: SkillInput) -> SkillOutput:
        start_time = time.time()

        if not AIOHTTP_AVAILABLE:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message="aiohttp library not available. Install with: pip install aiohttp",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=["Missing dependency: aiohttp"]
            )

        try:
            result = await self._make_http_request(skill_input.operation, skill_input.parameters)
            return SkillOutput(
                status=SkillStatus.SUCCESS,
                result=result,
                message=f"HTTP {skill_input.operation.upper()} completed successfully",
                execution_time_ms=(time.time() - start_time) * 1000,
                details=result
            )
        except Exception as e:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message=f"HTTP request failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )

    def _build_auth_header(self, auth: dict[str, Any] | None) -> dict[str, str]:
        """Build authentication headers."""
        if not auth:
            return {}

        auth_type = auth.get("type", "").lower()

        if auth_type == "basic":
            import base64
            username = auth.get("username", "")
            password = auth.get("password", "")
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            return {"Authorization": f"Basic {credentials}"}

        elif auth_type == "bearer":
            token = auth.get("token", "")
            return {"Authorization": f"Bearer {token}"}

        elif auth_type == "api_key":
            key_name = auth.get("key_name", "X-API-Key")
            key_value = auth.get("key_value", "")
            return {key_name: key_value}

        return {}

    async def _make_http_request(self, method: str, params: dict[str, Any]):
        """Make real HTTP request using aiohttp."""
        url = params["url"]
        headers = params.get("headers", {})
        timeout_seconds = params.get("timeout", 30)
        max_retries = params.get("max_retries", 0)

        # Add authentication headers
        auth = params.get("auth")
        if auth:
            auth_headers = self._build_auth_header(auth)
            headers.update(auth_headers)

        # Prepare request kwargs
        request_kwargs = {
            "headers": headers,
            "timeout": aiohttp.ClientTimeout(total=timeout_seconds)
        }

        # Add body data if present
        if "json" in params:
            request_kwargs["json"] = params["json"]
        elif "data" in params:
            request_kwargs["data"] = params["data"]

        # Add query parameters
        if "params" in params:
            request_kwargs["params"] = params["params"]

        # Retry logic
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(method.upper(), url, **request_kwargs) as response:
                        # Parse response
                        content_type = response.headers.get("Content-Type", "")

                        if "application/json" in content_type:
                            body = await response.json()
                        else:
                            body = await response.text()

                        return {
                            "method": method.upper(),
                            "url": str(response.url),
                            "status_code": response.status,
                            "headers": dict(response.headers),
                            "body": body,
                            "content_type": content_type,
                            "success": 200 <= response.status < 300,
                            "attempt": attempt + 1
                        }

            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    raise

        # Should not reach here
        raise last_exception or Exception("Request failed")


from hololoom.skills.base import register_skill

register_skill(RESTAPIClientSkill())
