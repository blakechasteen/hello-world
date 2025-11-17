"""
Comprehensive API Tests
"""

import pytest
from fastapi.testclient import TestClient
import tempfile
import shutil
from pathlib import Path

# Import the app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.main import app
from api.middleware.auth import init_default_auth
from promptly import Promptly


@pytest.fixture
def client():
    """Create test client"""
    # Initialize default auth
    api_key = init_default_auth()

    # Create test client
    with TestClient(app) as test_client:
        # Set default API key header
        test_client.headers.update({"X-API-Key": api_key})
        yield test_client


@pytest.fixture
def temp_promptly_dir():
    """Create temporary Promptly directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def promptly_instance(temp_promptly_dir):
    """Create Promptly instance"""
    p = Promptly(root_dir=temp_promptly_dir)
    p.init()
    return p


class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_health_check(self, client):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestPromptEndpoints:
    """Test prompt API endpoints"""

    def test_create_prompt(self, client, promptly_instance):
        """Test creating a prompt"""
        response = client.post(
            "/api/v1/prompts",
            json={
                "name": "test_prompt",
                "content": "This is a test prompt: {input}",
                "metadata": {"tags": ["test"]}
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"
        assert "test_prompt" in data["message"]

    def test_get_prompt(self, client, promptly_instance):
        """Test getting a prompt"""
        # First create a prompt
        promptly_instance.add("test_prompt", "Test content")

        # Get the prompt
        response = client.get("/api/v1/prompts/test_prompt")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "test_prompt"
        assert data["content"] == "Test content"
        assert "version" in data
        assert "commit_hash" in data

    def test_get_nonexistent_prompt(self, client, promptly_instance):
        """Test getting a prompt that doesn't exist"""
        response = client.get("/api/v1/prompts/nonexistent")
        assert response.status_code == 404

    def test_list_prompts(self, client, promptly_instance):
        """Test listing prompts"""
        # Create some prompts
        promptly_instance.add("prompt1", "Content 1")
        promptly_instance.add("prompt2", "Content 2")

        # List prompts
        response = client.get("/api/v1/prompts")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 2
        assert any(p["name"] == "prompt1" for p in data)
        assert any(p["name"] == "prompt2" for p in data)

    def test_search_prompts(self, client, promptly_instance):
        """Test searching prompts"""
        # Create prompts with metadata
        promptly_instance.add("prompt1", "AI summary", {"tags": ["ai", "nlp"]})
        promptly_instance.add("prompt2", "Text analysis", {"tags": ["nlp"]})

        # Search by tag
        response = client.post(
            "/api/v1/prompts/search",
            json={"tags": ["nlp"], "limit": 10}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1

    def test_get_prompt_diff(self, client, promptly_instance):
        """Test getting diff between prompt versions"""
        # Create two versions
        promptly_instance.add("test_prompt", "Version 1 content")
        promptly_instance.add("test_prompt", "Version 2 content")

        # Get diff
        response = client.get(
            "/api/v1/prompts/test_prompt/diff",
            params={"version_old": 1, "version_new": 2}
        )

        assert response.status_code == 200
        data = response.json()
        assert "diff" in data
        assert "changes_summary" in data


class TestBranchEndpoints:
    """Test branch API endpoints"""

    def test_create_branch(self, client, promptly_instance):
        """Test creating a branch"""
        response = client.post(
            "/api/v1/branches",
            json={"name": "feature/test", "from_branch": "main"}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"

    def test_list_branches(self, client, promptly_instance):
        """Test listing branches"""
        response = client.get("/api/v1/branches")
        assert response.status_code == 200

        data = response.json()
        assert "branches" in data
        assert "current_branch" in data
        assert len(data["branches"]) >= 1  # At least 'main' branch

    def test_checkout_branch(self, client, promptly_instance):
        """Test checking out a branch"""
        # Create a branch first
        promptly_instance.branch("test-branch")

        # Checkout the branch
        response = client.post(
            "/api/v1/branches/checkout",
            json={"branch_name": "test-branch"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_get_branch(self, client, promptly_instance):
        """Test getting branch details"""
        response = client.get("/api/v1/branches/main")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "main"
        assert "head_commit" in data


class TestEvaluationEndpoints:
    """Test evaluation API endpoints"""

    def test_run_evaluation(self, client, promptly_instance):
        """Test running an evaluation"""
        # Create a prompt first
        promptly_instance.add("test_prompt", "Summarize: {text}")

        # Run evaluation
        response = client.post(
            "/api/v1/evaluations",
            json={
                "prompt_name": "test_prompt",
                "test_cases": [
                    {
                        "inputs": {"text": "Long text here"},
                        "expected": "Summary"
                    }
                ],
                "evaluator": "keyword"
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert "evaluation_id" in data
        assert "results" in data
        assert "average_score" in data

    def test_get_evaluation(self, client, promptly_instance):
        """Test getting evaluation results"""
        # Create and run an evaluation
        promptly_instance.add("test_prompt", "Test: {input}")

        eval_response = client.post(
            "/api/v1/evaluations",
            json={
                "prompt_name": "test_prompt",
                "test_cases": [{"inputs": {"input": "test"}}],
                "evaluator": "keyword"
            }
        )

        eval_id = eval_response.json()["evaluation_id"]

        # Get evaluation
        response = client.get(f"/api/v1/evaluations/{eval_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["evaluation_id"] == eval_id

    def test_list_evaluations(self, client, promptly_instance):
        """Test listing evaluations"""
        response = client.get("/api/v1/evaluations")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestChainEndpoints:
    """Test chain API endpoints"""

    def test_create_chain(self, client, promptly_instance):
        """Test creating a chain"""
        # Create prompts for the chain
        promptly_instance.add("step1", "Step 1: {input}")
        promptly_instance.add("step2", "Step 2: {output}")

        # Create chain
        response = client.post(
            "/api/v1/chains",
            json={
                "name": "test_chain",
                "steps": ["step1", "step2"],
                "description": "Test chain"
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"

    def test_list_chains(self, client, promptly_instance):
        """Test listing chains"""
        response = client.get("/api/v1/chains")
        assert response.status_code == 200

        data = response.json()
        assert "chains" in data
        assert "total" in data

    def test_execute_chain(self, client, promptly_instance):
        """Test executing a chain"""
        # Create prompts and chain
        promptly_instance.add("step1", "Process: {input}")
        promptly_instance.create_chain("test_chain", ["step1"])

        # Execute chain
        response = client.post(
            "/api/v1/chains/execute",
            json={
                "chain_name": "test_chain",
                "initial_input": {"input": "test"}
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "execution_id" in data
        assert "steps" in data


class TestPluginEndpoints:
    """Test plugin API endpoints"""

    def test_list_plugins(self, client):
        """Test listing plugins"""
        response = client.get("/api/v1/plugins")
        assert response.status_code == 200

        data = response.json()
        assert "evaluators" in data
        assert "storage_backends" in data
        assert "total" in data


class TestAuthentication:
    """Test authentication"""

    def test_missing_api_key(self):
        """Test request without API key"""
        client = TestClient(app)
        response = client.get("/api/v1/prompts")
        assert response.status_code == 401

    def test_invalid_api_key(self):
        """Test request with invalid API key"""
        client = TestClient(app)
        client.headers.update({"X-API-Key": "invalid-key"})
        response = client.get("/api/v1/prompts")
        assert response.status_code == 401


class TestRateLimiting:
    """Test rate limiting"""

    def test_rate_limit(self, client):
        """Test rate limiting (simplified test)"""
        # Make many requests quickly
        # Note: This might not trigger rate limit in test environment
        # depending on configuration
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code in [200, 429]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
