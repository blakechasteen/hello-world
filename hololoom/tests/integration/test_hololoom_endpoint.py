#!/usr/bin/env python3
"""Integration checks for the agentic FastAPI endpoint."""

from __future__ import annotations

import json
import os
import time
from typing import Iterable

import pytest
import requests


def _candidate_base_urls() -> Iterable[str]:
    primary = os.environ.get("HOLOLOOM_ENDPOINT", "http://localhost:8765")
    yield primary
    # Common fallback when running via start_agentic_server.py
    if primary != "http://localhost:8001":
        yield "http://localhost:8001"


def _select_base_url() -> str:
    failures = []
    for base in _candidate_base_urls():
        try:
            response = requests.get(f"{base}/health", timeout=4)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - diagnostic path
            failures.append(f"{base} ({exc})")
            continue
        return base
    pytest.skip("Agentic server not reachable: " + "; ".join(failures))


BASE_URL = _select_base_url()


def test_health_endpoint() -> None:
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    response.raise_for_status()
    payload = response.json()
    assert payload
    assert payload.get("status") in {"ok", "online", "ready"}


def test_weaving_endpoint_completes() -> None:
    weaving_request = {
        "query": "What is Thompson Sampling?",
        "pattern": "fast",
        "complexity": "auto",
    }

    response = requests.post(
        f"{BASE_URL}/execute/hololoom",
        json=weaving_request,
        timeout=20,
    )
    response.raise_for_status()
    result = response.json()
    execution_id = result.get("execution_id")
    assert execution_id, f"Unexpected response: {json.dumps(result, indent=2)}"

    for _ in range(20):
        time.sleep(0.5)
        status_response = requests.get(
            f"{BASE_URL}/execute/status/{execution_id}",
            timeout=20,
        )
        status_response.raise_for_status()
        status_payload = status_response.json()
        state = status_payload.get("status", "")
        if state in {"completed", "failed"}:
            assert state == "completed", json.dumps(status_payload, indent=2)
            return
    pytest.fail("HoloLoom execution did not complete within the allotted time")
