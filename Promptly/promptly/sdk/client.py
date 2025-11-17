"""
Promptly Synchronous Client SDK
"""

import requests
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin
import time

from .exceptions import (
    AuthenticationError,
    PromptNotFoundError,
    BranchNotFoundError,
    RateLimitError,
    APIError
)


class PromptlyClient:
    """
    Synchronous client for Promptly API

    Example:
        >>> client = PromptlyClient(base_url="http://localhost:8000", api_key="your-api-key")
        >>> prompts = client.list_prompts()
        >>> prompt = client.get_prompt("summarizer")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Promptly client

        Args:
            base_url: Base URL of Promptly API
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Dict = None,
        params: Dict = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        url = urljoin(self.base_url, endpoint)

        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=self.timeout
            )

            # Handle rate limiting with retry
            if response.status_code == 429:
                if retry_count < self.max_retries:
                    retry_after = int(response.headers.get("Retry-After", self.retry_delay))
                    time.sleep(retry_after)
                    return self._request(method, endpoint, data, params, retry_count + 1)
                raise RateLimitError("Rate limit exceeded")

            # Handle authentication errors
            if response.status_code == 401:
                raise AuthenticationError("Authentication failed. Check your API key.")

            # Handle not found errors
            if response.status_code == 404:
                error_data = response.json() if response.text else {}
                error_msg = error_data.get("detail", "Resource not found")

                if "Prompt" in error_msg:
                    raise PromptNotFoundError(error_msg)
                elif "Branch" in error_msg:
                    raise BranchNotFoundError(error_msg)
                else:
                    raise APIError(error_msg, status_code=404, response=error_data)

            # Handle other errors
            if not response.ok:
                error_data = response.json() if response.text else {}
                error_msg = error_data.get("detail") or error_data.get("message", "API request failed")
                raise APIError(error_msg, status_code=response.status_code, response=error_data)

            # Return JSON response
            return response.json() if response.text else {}

        except requests.exceptions.Timeout:
            if retry_count < self.max_retries:
                time.sleep(self.retry_delay)
                return self._request(method, endpoint, data, params, retry_count + 1)
            raise APIError("Request timeout")

        except requests.exceptions.ConnectionError:
            if retry_count < self.max_retries:
                time.sleep(self.retry_delay)
                return self._request(method, endpoint, data, params, retry_count + 1)
            raise APIError("Connection error")

    # Prompt methods

    def create_prompt(self, name: str, content: str, metadata: Dict = None) -> Dict:
        """Create or update a prompt"""
        data = {
            "name": name,
            "content": content,
            "metadata": metadata
        }
        return self._request("POST", "/api/v1/prompts", data=data)

    def get_prompt(self, name: str, version: int = None, commit_hash: str = None) -> Dict:
        """Get a prompt by name"""
        params = {}
        if version:
            params["version"] = version
        if commit_hash:
            params["commit_hash"] = commit_hash

        return self._request("GET", f"/api/v1/prompts/{name}", params=params)

    def list_prompts(self, branch: str = None) -> List[Dict]:
        """List all prompts"""
        params = {"branch": branch} if branch else {}
        return self._request("GET", "/api/v1/prompts", params=params)

    def search_prompts(
        self,
        query: str = None,
        tags: List[str] = None,
        branch: str = None,
        limit: int = 20
    ) -> List[Dict]:
        """Search prompts"""
        data = {
            "query": query,
            "tags": tags,
            "branch": branch,
            "limit": limit
        }
        return self._request("POST", "/api/v1/prompts/search", data=data)

    def get_prompt_diff(self, name: str, version_old: int, version_new: int) -> Dict:
        """Get diff between two prompt versions"""
        params = {
            "version_old": version_old,
            "version_new": version_new
        }
        return self._request("GET", f"/api/v1/prompts/{name}/diff", params=params)

    # Branch methods

    def create_branch(self, name: str, from_branch: str = None) -> Dict:
        """Create a new branch"""
        data = {
            "name": name,
            "from_branch": from_branch
        }
        return self._request("POST", "/api/v1/branches", data=data)

    def checkout_branch(self, branch_name: str) -> Dict:
        """Checkout a branch"""
        data = {"branch_name": branch_name}
        return self._request("POST", "/api/v1/branches/checkout", data=data)

    def list_branches(self) -> Dict:
        """List all branches"""
        return self._request("GET", "/api/v1/branches")

    def get_branch(self, branch_name: str) -> Dict:
        """Get branch details"""
        return self._request("GET", f"/api/v1/branches/{branch_name}")

    def delete_branch(self, branch_name: str, force: bool = False) -> Dict:
        """Delete a branch"""
        params = {"force": force}
        return self._request("DELETE", f"/api/v1/branches/{branch_name}", params=params)

    # History methods

    def get_log(self, name: str = None, limit: int = 10) -> List[Dict]:
        """Get commit history"""
        params = {"limit": limit}
        if name:
            params["name"] = name
        return self._request("GET", "/api/v1/history/log", params=params)

    def get_blame(self, name: str, version: int = None) -> List[Dict]:
        """Get blame information for a prompt"""
        params = {"version": version} if version else {}
        return self._request("GET", f"/api/v1/history/blame/{name}", params=params)

    # Evaluation methods

    def run_evaluation(
        self,
        prompt_name: str,
        test_cases: List[Dict],
        evaluator: str = "keyword",
        model_endpoint: str = None
    ) -> Dict:
        """Run evaluation on a prompt"""
        data = {
            "prompt_name": prompt_name,
            "test_cases": test_cases,
            "evaluator": evaluator,
            "model_endpoint": model_endpoint
        }
        return self._request("POST", "/api/v1/evaluations", data=data)

    def get_evaluation(self, evaluation_id: str) -> Dict:
        """Get evaluation results"""
        return self._request("GET", f"/api/v1/evaluations/{evaluation_id}")

    def list_evaluations(self, prompt_name: str = None, limit: int = 20) -> List[Dict]:
        """List evaluations"""
        params = {"limit": limit}
        if prompt_name:
            params["prompt_name"] = prompt_name
        return self._request("GET", "/api/v1/evaluations", params=params)

    def compare_evaluations(self, evaluation_ids: List[str]) -> Dict:
        """Compare evaluations"""
        data = {"evaluation_ids": evaluation_ids}
        return self._request("POST", "/api/v1/evaluations/compare", data=data)

    # Chain methods

    def create_chain(self, name: str, steps: List[str], description: str = None) -> Dict:
        """Create a new chain"""
        data = {
            "name": name,
            "steps": steps,
            "description": description
        }
        return self._request("POST", "/api/v1/chains", data=data)

    def list_chains(self) -> Dict:
        """List all chains"""
        return self._request("GET", "/api/v1/chains")

    def get_chain(self, chain_name: str) -> Dict:
        """Get chain details"""
        return self._request("GET", f"/api/v1/chains/{chain_name}")

    def execute_chain(
        self,
        chain_name: str,
        initial_input: Dict,
        model_endpoint: str = None,
        stream: bool = False
    ) -> Dict:
        """Execute a chain"""
        data = {
            "chain_name": chain_name,
            "initial_input": initial_input,
            "model_endpoint": model_endpoint,
            "stream": stream
        }
        return self._request("POST", "/api/v1/chains/execute", data=data)

    def get_execution_status(self, execution_id: str) -> Dict:
        """Get chain execution status"""
        return self._request("GET", f"/api/v1/chains/executions/{execution_id}")

    def delete_chain(self, chain_name: str) -> Dict:
        """Delete a chain"""
        return self._request("DELETE", f"/api/v1/chains/{chain_name}")

    # Plugin methods

    def list_plugins(self) -> Dict:
        """List all available plugins"""
        return self._request("GET", "/api/v1/plugins")

    def get_plugin(self, plugin_type: str, plugin_name: str) -> Dict:
        """Get plugin details"""
        return self._request("GET", f"/api/v1/plugins/{plugin_type}/{plugin_name}")

    # Health check

    def health(self) -> Dict:
        """Check API health"""
        return self._request("GET", "/health")

    def close(self):
        """Close the session"""
        self.session.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
