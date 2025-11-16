#!/usr/bin/env python3
"""
API Connector Module
====================
Connect to external APIs, fetch documentation, and use as context for code generation.

Capabilities:
- OpenAPI/Swagger spec parsing
- REST API documentation fetching
- GraphQL schema introspection
- API endpoint extraction
- Example generation
- Code generation with API context

Author: Claude Code
Date: November 16, 2025
"""

import logging
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class APIType(Enum):
    """Supported API types"""
    OPENAPI = "openapi"
    SWAGGER = "swagger"
    GRAPHQL = "graphql"
    REST = "rest"
    UNKNOWN = "unknown"


@dataclass
class APIEndpoint:
    """API endpoint information"""
    path: str
    method: str  # GET, POST, PUT, DELETE, etc.
    description: Optional[str] = None
    parameters: List[Dict[str, Any]] = None
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Any] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []
        if self.responses is None:
            self.responses = {}
        if self.tags is None:
            self.tags = []


@dataclass
class APIMetadata:
    """Metadata about connected API"""
    title: str
    version: str
    base_url: str
    api_type: str
    total_endpoints: int
    endpoints_by_tag: Dict[str, int]
    timestamp: str


class APIConnector:
    """
    Connector for external APIs.

    Fetches documentation, parses schemas, creates context for code generation.
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize API connector.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.stats = {
            "apis_connected": 0,
            "endpoints_extracted": 0,
            "errors": 0
        }

    def connect_openapi(
        self,
        spec_url: str,
        headers: Optional[Dict[str, str]] = None
    ) -> tuple[List[APIEndpoint], APIMetadata]:
        """
        Connect to API via OpenAPI/Swagger spec.

        Args:
            spec_url: URL to OpenAPI spec (JSON or YAML)
            headers: Optional HTTP headers

        Returns:
            Tuple of (endpoints, metadata)
        """
        logger.info(f"Connecting to OpenAPI spec: {spec_url}")

        try:
            # Fetch spec
            response = requests.get(
                spec_url,
                headers=headers or {},
                timeout=self.timeout
            )
            response.raise_for_status()

            spec = response.json()

            # Parse spec
            endpoints = self._parse_openapi_spec(spec)

            # Build metadata
            info = spec.get("info", {})
            servers = spec.get("servers", [])
            base_url = servers[0].get("url", "") if servers else ""

            # Count endpoints by tag
            tags_count = {}
            for endpoint in endpoints:
                for tag in endpoint.tags:
                    tags_count[tag] = tags_count.get(tag, 0) + 1

            metadata = APIMetadata(
                title=info.get("title", "Unknown API"),
                version=info.get("version", "unknown"),
                base_url=base_url,
                api_type=APIType.OPENAPI.value,
                total_endpoints=len(endpoints),
                endpoints_by_tag=tags_count,
                timestamp=datetime.now().isoformat()
            )

            self.stats["apis_connected"] += 1
            self.stats["endpoints_extracted"] += len(endpoints)

            logger.info(f"Connected to {metadata.title} - {len(endpoints)} endpoints")
            return endpoints, metadata

        except Exception as e:
            logger.error(f"Failed to connect to OpenAPI spec: {e}")
            self.stats["errors"] += 1
            raise

    def _parse_openapi_spec(self, spec: Dict[str, Any]) -> List[APIEndpoint]:
        """Parse OpenAPI/Swagger spec and extract endpoints"""
        endpoints = []

        paths = spec.get("paths", {})

        for path, path_item in paths.items():
            # Each path can have multiple methods
            for method in ["get", "post", "put", "delete", "patch", "options", "head"]:
                if method not in path_item:
                    continue

                operation = path_item[method]

                # Extract parameters
                parameters = []
                for param in operation.get("parameters", []):
                    parameters.append({
                        "name": param.get("name"),
                        "in": param.get("in"),  # query, path, header, cookie
                        "required": param.get("required", False),
                        "type": param.get("type") or param.get("schema", {}).get("type"),
                        "description": param.get("description")
                    })

                # Extract request body (OpenAPI 3.0)
                request_body = None
                if "requestBody" in operation:
                    content = operation["requestBody"].get("content", {})
                    if "application/json" in content:
                        schema = content["application/json"].get("schema", {})
                        request_body = schema

                # Extract responses
                responses = {}
                for status_code, response in operation.get("responses", {}).items():
                    responses[status_code] = {
                        "description": response.get("description"),
                        "schema": response.get("schema") or response.get("content", {}).get("application/json", {}).get("schema")
                    }

                endpoint = APIEndpoint(
                    path=path,
                    method=method.upper(),
                    description=operation.get("summary") or operation.get("description"),
                    parameters=parameters,
                    request_body=request_body,
                    responses=responses,
                    tags=operation.get("tags", [])
                )

                endpoints.append(endpoint)

        return endpoints

    def connect_graphql(
        self,
        graphql_url: str,
        headers: Optional[Dict[str, str]] = None
    ) -> tuple[Dict[str, Any], APIMetadata]:
        """
        Connect to GraphQL API via introspection.

        Args:
            graphql_url: GraphQL endpoint URL
            headers: Optional HTTP headers

        Returns:
            Tuple of (schema, metadata)
        """
        logger.info(f"Connecting to GraphQL API: {graphql_url}")

        try:
            # Introspection query
            introspection_query = """
            query IntrospectionQuery {
                __schema {
                    queryType { name }
                    mutationType { name }
                    types {
                        name
                        kind
                        description
                        fields {
                            name
                            description
                            args {
                                name
                                type { name }
                            }
                        }
                    }
                }
            }
            """

            response = requests.post(
                graphql_url,
                json={"query": introspection_query},
                headers=headers or {},
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            schema = data.get("data", {}).get("__schema", {})

            # Build metadata
            types = schema.get("types", [])
            query_type = schema.get("queryType", {}).get("name", "Query")
            mutation_type = schema.get("mutationType", {}).get("name", "Mutation")

            metadata = APIMetadata(
                title="GraphQL API",
                version="unknown",
                base_url=graphql_url,
                api_type=APIType.GRAPHQL.value,
                total_endpoints=len(types),
                endpoints_by_tag={"queries": 1, "mutations": 1},
                timestamp=datetime.now().isoformat()
            )

            self.stats["apis_connected"] += 1

            logger.info(f"Connected to GraphQL API - {len(types)} types")
            return schema, metadata

        except Exception as e:
            logger.error(f"Failed to connect to GraphQL API: {e}")
            self.stats["errors"] += 1
            raise

    def connect_rest_api(
        self,
        base_url: str,
        endpoints: List[Dict[str, str]],
        headers: Optional[Dict[str, str]] = None
    ) -> tuple[List[APIEndpoint], APIMetadata]:
        """
        Connect to REST API with manually specified endpoints.

        Args:
            base_url: Base URL of API
            endpoints: List of endpoint dicts with path, method, description
            headers: Optional HTTP headers

        Returns:
            Tuple of (endpoints, metadata)
        """
        logger.info(f"Connecting to REST API: {base_url}")

        api_endpoints = []
        for ep in endpoints:
            endpoint = APIEndpoint(
                path=ep.get("path", ""),
                method=ep.get("method", "GET"),
                description=ep.get("description"),
                tags=[ep.get("tag", "default")]
            )
            api_endpoints.append(endpoint)

        metadata = APIMetadata(
            title="REST API",
            version="unknown",
            base_url=base_url,
            api_type=APIType.REST.value,
            total_endpoints=len(api_endpoints),
            endpoints_by_tag={},
            timestamp=datetime.now().isoformat()
        )

        self.stats["apis_connected"] += 1
        self.stats["endpoints_extracted"] += len(api_endpoints)

        return api_endpoints, metadata

    def endpoints_to_memory_shards(
        self,
        endpoints: List[APIEndpoint],
        metadata: APIMetadata
    ) -> List[Dict[str, Any]]:
        """
        Convert API endpoints to HoloLoom MemoryShard format.

        Returns list of shard dictionaries.
        """
        shards = []

        for endpoint in endpoints:
            # Create text representation
            text_parts = [
                f"API Endpoint: {endpoint.method} {endpoint.path}",
                f"API: {metadata.title} v{metadata.version}"
            ]

            if endpoint.description:
                text_parts.append(f"Description: {endpoint.description}")

            if endpoint.parameters:
                params = [f"{p['name']} ({p.get('type', 'unknown')})" for p in endpoint.parameters[:3]]
                text_parts.append(f"Parameters: {', '.join(params)}")

            if endpoint.request_body:
                text_parts.append("Accepts JSON request body")

            if endpoint.responses:
                status_codes = list(endpoint.responses.keys())
                text_parts.append(f"Responses: {', '.join(status_codes[:3])}")

            text = "\n".join(text_parts)

            # Create shard
            shard = {
                "id": f"api-{metadata.title.replace(' ', '-')}-{endpoint.method}-{endpoint.path.replace('/', '-')}",
                "text": text,
                "entities": [metadata.title, endpoint.method, endpoint.path],
                "motifs": ["api", "endpoint", metadata.api_type] + endpoint.tags,
                "metadata": {
                    "source": "api_connector",
                    "api_title": metadata.title,
                    "api_version": metadata.version,
                    "api_type": metadata.api_type,
                    "base_url": metadata.base_url,
                    "endpoint_path": endpoint.path,
                    "endpoint_method": endpoint.method,
                    "timestamp": datetime.now().isoformat()
                }
            }

            shards.append(shard)

        return shards

    def generate_api_client_code(
        self,
        endpoints: List[APIEndpoint],
        metadata: APIMetadata,
        language: str = "python"
    ) -> str:
        """
        Generate API client code from endpoints.

        Args:
            endpoints: List of API endpoints
            metadata: API metadata
            language: Target language (python, typescript, etc.)

        Returns:
            Generated client code
        """
        if language == "python":
            return self._generate_python_client(endpoints, metadata)
        elif language in ["typescript", "javascript"]:
            return self._generate_typescript_client(endpoints, metadata)
        else:
            return f"# API client generation not yet supported for {language}"

    def _generate_python_client(
        self,
        endpoints: List[APIEndpoint],
        metadata: APIMetadata
    ) -> str:
        """Generate Python API client code"""
        class_name = metadata.title.replace(" ", "").replace("-", "") + "Client"

        code = f'''"""
{metadata.title} API Client
Generated from API spec
"""

import requests
from typing import Dict, Any, Optional


class {class_name}:
    """Client for {metadata.title} v{metadata.version}"""

    def __init__(self, base_url: str = "{metadata.base_url}", api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()

        if api_key:
            self.session.headers.update({{"Authorization": f"Bearer {{api_key}}"}})

'''

        # Generate methods for each endpoint
        for endpoint in endpoints[:10]:  # Limit to first 10
            method_name = endpoint.path.replace('/', '_').replace('{', '').replace('}', '').strip('_')
            method_name = f"{endpoint.method.lower()}_{method_name}"

            # Generate docstring
            docstring = endpoint.description or f"{endpoint.method} {endpoint.path}"

            code += f'''    def {method_name}(self'''

            # Add parameters
            if endpoint.parameters:
                for param in endpoint.parameters[:3]:
                    param_name = param["name"]
                    is_required = param.get("required", False)
                    if is_required:
                        code += f', {param_name}'
                    else:
                        code += f', {param_name}=None'

            code += f'''):
        """
        {docstring}
        """
        response = self.session.{endpoint.method.lower()}(
            f"{{self.base_url}}{endpoint.path}"
        )
        response.raise_for_status()
        return response.json()

'''

        return code

    def _generate_typescript_client(
        self,
        endpoints: List[APIEndpoint],
        metadata: APIMetadata
    ) -> str:
        """Generate TypeScript API client code"""
        class_name = metadata.title.replace(" ", "").replace("-", "") + "Client"

        code = f'''/**
 * {metadata.title} API Client
 * Generated from API spec
 */

export class {class_name} {{
    private baseUrl: string;
    private apiKey?: string;

    constructor(baseUrl: string = "{metadata.base_url}", apiKey?: string) {{
        this.baseUrl = baseUrl;
        this.apiKey = apiKey;
    }}

'''

        # Generate methods for each endpoint
        for endpoint in endpoints[:10]:
            method_name = endpoint.path.replace('/', '_').replace('{', '').replace('}', '').strip('_')
            method_name = f"{endpoint.method.lower()}_{method_name}"

            docstring = endpoint.description or f"{endpoint.method} {endpoint.path}"

            code += f'''    /**
     * {docstring}
     */
    async {method_name}(): Promise<any> {{
        const response = await fetch(`${{this.baseUrl}}{endpoint.path}`, {{
            method: '{endpoint.method}',
            headers: this.apiKey ? {{ 'Authorization': `Bearer ${{this.apiKey}}` }} : {{}}
        }});

        if (!response.ok) {{
            throw new Error(`API error: ${{response.statusText}}`);
        }}

        return response.json();
    }}

'''

        code += '}\n'
        return code

    def get_stats(self) -> Dict[str, int]:
        """Get connector statistics"""
        return self.stats.copy()
