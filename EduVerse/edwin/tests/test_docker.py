"""
EdWIN Docker Tests

Tests for Docker configuration and build.

Implementation Date: November 15, 2025

Usage:
    pytest EduVerse/edwin/tests/test_docker.py -v
"""

import pytest
import subprocess
import os
from pathlib import Path


# Get repository root
REPO_ROOT = Path(__file__).parent.parent.parent.parent


# ==============================================================================
# Docker File Tests
# ==============================================================================

def test_dockerfile_exists():
    """Test that Dockerfile exists"""
    dockerfile = REPO_ROOT / "Dockerfile"
    assert dockerfile.exists(), "Dockerfile not found"


def test_dockerignore_exists():
    """Test that .dockerignore exists"""
    dockerignore = REPO_ROOT / ".dockerignore"
    assert dockerignore.exists(), ".dockerignore not found"


def test_docker_compose_exists():
    """Test that docker-compose files exist"""
    compose_file = REPO_ROOT / "docker-compose.edwin.yml"
    assert compose_file.exists(), "docker-compose.edwin.yml not found"


# ==============================================================================
# Docker Build Tests (Optional - slow)
# ==============================================================================

@pytest.mark.slow
@pytest.mark.integration
def test_docker_build():
    """Test Docker image build (slow test)"""
    try:
        result = subprocess.run(
            ["docker", "build", "-t", "edwin-test:latest", "."],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        assert result.returncode == 0, f"Docker build failed: {result.stderr}"

    except subprocess.TimeoutExpired:
        pytest.fail("Docker build timed out after 10 minutes")
    except FileNotFoundError:
        pytest.skip("Docker not installed")


@pytest.mark.slow
@pytest.mark.integration
def test_docker_compose_config():
    """Test docker-compose configuration is valid"""
    compose_file = REPO_ROOT / "docker-compose.edwin.yml"

    try:
        result = subprocess.run(
            ["docker-compose", "-f", str(compose_file), "config"],
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode == 0, f"docker-compose config invalid: {result.stderr}"

    except FileNotFoundError:
        pytest.skip("docker-compose not installed")


# ==============================================================================
# Dockerfile Content Tests
# ==============================================================================

def test_dockerfile_uses_multi_stage_build():
    """Test that Dockerfile uses multi-stage build"""
    dockerfile = REPO_ROOT / "Dockerfile"

    with open(dockerfile, 'r') as f:
        content = f.read()

    # Check for multiple FROM statements (multi-stage)
    from_count = content.count("FROM")
    assert from_count >= 2, "Dockerfile should use multi-stage build (multiple FROM)"


def test_dockerfile_uses_non_root_user():
    """Test that Dockerfile runs as non-root user"""
    dockerfile = REPO_ROOT / "Dockerfile"

    with open(dockerfile, 'r') as f:
        content = f.read()

    assert "USER edwin" in content, "Dockerfile should run as non-root user"


def test_dockerfile_has_healthcheck():
    """Test that Dockerfile includes health check"""
    dockerfile = REPO_ROOT / "Dockerfile"

    with open(dockerfile, 'r') as f:
        content = f.read()

    assert "HEALTHCHECK" in content, "Dockerfile should include HEALTHCHECK"


# ==============================================================================
# Docker Compose Tests
# ==============================================================================

def test_docker_compose_has_required_services():
    """Test that docker-compose has all required services"""
    compose_file = REPO_ROOT / "docker-compose.edwin.yml"

    with open(compose_file, 'r') as f:
        content = f.read()

    required_services = [
        "edwin-api",
        "edwin-dashboard",
        "edwin-mobile",
        "neo4j",
        "qdrant",
        "redis"
    ]

    for service in required_services:
        assert service in content, f"docker-compose missing service: {service}"


def test_docker_compose_has_health_checks():
    """Test that services have health checks"""
    compose_file = REPO_ROOT / "docker-compose.edwin.yml"

    with open(compose_file, 'r') as f:
        content = f.read()

    # Should have health checks for all services
    assert content.count("healthcheck:") >= 6, "Missing health checks for services"


def test_docker_compose_uses_volumes():
    """Test that docker-compose uses volumes for persistence"""
    compose_file = REPO_ROOT / "docker-compose.edwin.yml"

    with open(compose_file, 'r') as f:
        content = f.read()

    assert "volumes:" in content, "docker-compose should define volumes"
    assert "neo4j-data" in content, "Missing Neo4j data volume"
    assert "qdrant-data" in content, "Missing Qdrant data volume"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
