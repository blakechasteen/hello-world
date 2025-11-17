"""
SDK Client Tests
"""

import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sdk.client import PromptlyClient
from sdk.exceptions import AuthenticationError, PromptNotFoundError, RateLimitError


class TestPromptlyClient:
    """Test synchronous client"""

    def test_client_initialization(self):
        """Test client initialization"""
        client = PromptlyClient(
            base_url="http://localhost:8000",
            api_key="test-key"
        )

        assert client.base_url == "http://localhost:8000"
        assert client.api_key == "test-key"
        assert "X-API-Key" in client.session.headers

    def test_context_manager(self):
        """Test client as context manager"""
        with PromptlyClient(api_key="test-key") as client:
            assert client.session is not None

    @patch('requests.Session.request')
    def test_create_prompt(self, mock_request):
        """Test creating a prompt"""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 201
        mock_response.text = '{"status": "success"}'
        mock_response.json.return_value = {"status": "success"}
        mock_request.return_value = mock_response

        client = PromptlyClient(api_key="test-key")
        result = client.create_prompt("test", "content")

        assert result["status"] == "success"

    @patch('requests.Session.request')
    def test_authentication_error(self, mock_request):
        """Test authentication error handling"""
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 401
        mock_response.text = '{"detail": "Unauthorized"}'
        mock_request.return_value = mock_response

        client = PromptlyClient(api_key="invalid-key")

        with pytest.raises(AuthenticationError):
            client.get_prompt("test")

    @patch('requests.Session.request')
    def test_rate_limit_error(self, mock_request):
        """Test rate limit error handling"""
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "1"}
        mock_request.return_value = mock_response

        client = PromptlyClient(api_key="test-key", max_retries=1)

        with pytest.raises(RateLimitError):
            client.get_prompt("test")

    @patch('requests.Session.request')
    def test_not_found_error(self, mock_request):
        """Test not found error handling"""
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 404
        mock_response.text = '{"detail": "Prompt not found"}'
        mock_response.json.return_value = {"detail": "Prompt not found"}
        mock_request.return_value = mock_response

        client = PromptlyClient(api_key="test-key")

        with pytest.raises(PromptNotFoundError):
            client.get_prompt("nonexistent")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
