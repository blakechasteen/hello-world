"""API interface — test HTTP APIs via httpx."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from .base import Action, Observation


@dataclass
class APIInterface:
    """Interact with an HTTP API."""

    base_url: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    openapi_spec: dict | None = None  # parsed OpenAPI spec, if available
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _history: list[dict] = field(default_factory=list, init=False)

    async def start(self) -> Observation:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=self.timeout,
        )

        parts = [f"API session ready. Base URL: {self.base_url}"]

        if self.openapi_spec:
            parts.append("")
            parts.append("OPENAPI SPEC SUMMARY:")
            parts.append(_summarize_openapi(self.openapi_spec))

        # Try a health check / root probe.
        try:
            resp = await self._client.get("/")
            parts.append("")
            parts.append(f"GET / → {resp.status_code}")
            parts.append(_format_response(resp))
        except Exception as e:
            parts.append(f"\nGET / failed: {e}")

        return Observation(content="\n".join(parts))

    async def act(self, action: Action) -> Observation:
        meta = action.metadata or {}
        method = action.kind.upper()
        path = action.value

        if method not in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
            return Observation(
                content=f"Unknown HTTP method: {method}. Use GET, POST, PUT, PATCH, DELETE, HEAD, or OPTIONS.",
                raw={"error": f"unknown method: {method}"},
            )

        # Build request kwargs.
        kwargs: dict[str, Any] = {}

        # JSON body.
        body = meta.get("body") or meta.get("json")
        if body:
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except json.JSONDecodeError:
                    pass
            kwargs["json"] = body

        # Query params.
        params = meta.get("params") or meta.get("query")
        if params:
            kwargs["params"] = params

        # Per-request headers.
        req_headers = meta.get("headers")
        if req_headers:
            kwargs["headers"] = req_headers

        # Form data.
        form = meta.get("form") or meta.get("data")
        if form:
            kwargs["data"] = form

        try:
            resp = await self._client.request(method, path, **kwargs)
        except Exception as e:
            return Observation(
                content=f"REQUEST FAILED: {method} {path} — {type(e).__name__}: {e}",
                raw={"error": str(e), "method": method, "path": path},
            )

        # Record in history.
        entry = {
            "method": method,
            "path": path,
            "status": resp.status_code,
            "body_preview": resp.text[:500] if resp.text else "",
        }
        self._history.append(entry)

        parts = [f"{method} {path} → {resp.status_code}"]
        parts.append(_format_response(resp))

        return Observation(
            content="\n".join(parts),
            raw={
                "status_code": resp.status_code,
                "headers": dict(resp.headers),
                "body": resp.text[:2000] if resp.text else "",
            },
        )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()


def _format_response(resp: httpx.Response, max_body: int = 2000) -> str:
    """Format an HTTP response for the agent to read."""
    parts = []

    # Headers (selective).
    interesting = ["content-type", "location", "www-authenticate", "x-request-id",
                   "retry-after", "x-ratelimit-remaining"]
    for h in interesting:
        if h in resp.headers:
            parts.append(f"  {h}: {resp.headers[h]}")

    # Body.
    ct = resp.headers.get("content-type", "")
    body = resp.text[:max_body] if resp.text else ""

    if "json" in ct:
        try:
            parsed = resp.json()
            body = json.dumps(parsed, indent=2)[:max_body]
        except Exception:
            pass
        parts.append(f"\nRESPONSE BODY (JSON):\n{body}")
    elif body:
        parts.append(f"\nRESPONSE BODY:\n{body}")
    else:
        parts.append("\n(empty body)")

    return "\n".join(parts)


def _summarize_openapi(spec: dict) -> str:
    """Produce a concise summary of an OpenAPI spec for the agent."""
    lines = []

    info = spec.get("info", {})
    if info.get("title"):
        lines.append(f"API: {info['title']} (v{info.get('version', '?')})")
    if info.get("description"):
        desc = info["description"][:200]
        lines.append(f"Description: {desc}")

    paths = spec.get("paths", {})
    lines.append(f"\nEndpoints ({len(paths)}):")

    for path, methods in sorted(paths.items()):
        for method in sorted(methods):
            if method in ("get", "post", "put", "patch", "delete"):
                op = methods[method]
                summary = op.get("summary", op.get("operationId", ""))
                params = op.get("parameters", [])
                param_names = [p.get("name", "") for p in params if p.get("name")]
                body_hint = ""
                if op.get("requestBody"):
                    body_hint = " [body]"

                line = f"  {method.upper():7s} {path}"
                if summary:
                    line += f"  — {summary}"
                if param_names:
                    line += f"  params: {', '.join(param_names)}"
                line += body_hint
                lines.append(line)

    # Auth schemes.
    security = spec.get("components", {}).get("securitySchemes", {})
    if security:
        lines.append(f"\nAuth schemes: {', '.join(security.keys())}")

    return "\n".join(lines)


async def load_openapi_spec(url_or_path: str) -> dict:
    """Load an OpenAPI spec from a URL or file path."""
    import os

    if os.path.isfile(url_or_path):
        with open(url_or_path) as f:
            text = f.read()
    else:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url_or_path)
            resp.raise_for_status()
            text = resp.text

    # Try JSON first, then YAML.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
            return yaml.safe_load(text)
        except Exception:
            raise ValueError(f"Could not parse spec as JSON or YAML: {url_or_path}")
