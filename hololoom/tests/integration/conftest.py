"""
Integration test configuration.

Auto-skips tests marked @requires_docker when Docker services aren't reachable.
"""

import os
import socket

import pytest


def _tcp_reachable(host: str, port: int, timeout: float = 0.2) -> bool:
    """Quick check whether a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def _docker_services_available() -> bool:
    """Check whether the required Docker services are reachable."""
    # Neo4j bolt default port + Postgres default port
    return _tcp_reachable("localhost", 7687) and _tcp_reachable("localhost", 5432)


def pytest_collection_modifyitems(config, items):
    """Skip @requires_docker tests when services aren't reachable."""
    if os.environ.get("HOLOLOOM_RUN_DOCKER_TESTS") == "1":
        return  # User explicitly opted in
    if _docker_services_available():
        return  # Services are up, run the tests
    skip_marker = pytest.mark.skip(reason="Docker services (neo4j, postgres) not reachable")
    for item in items:
        if "requires_docker" in item.keywords:
            item.add_marker(skip_marker)
