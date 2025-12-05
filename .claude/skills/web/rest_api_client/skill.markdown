# Skill: REST API Client

## Metadata

- **Name**: `rest_api_client`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-24`
- **Last Updated**: `2025-11-24`
- **Category**: `web`
- **Tags**: `http, rest, api, requests, web, json`

## Description

**Short Description**:
Generic HTTP/REST API client for web service integration with full CRUD support.

**Detailed Description**:
The REST API Client skill provides comprehensive HTTP request capabilities for integrating with web services and REST APIs. Supports all HTTP methods (GET, POST, PUT, PATCH, DELETE, HEAD) with custom headers, authentication (Bearer, Basic, API keys), request body encoding (JSON, form data, multipart), query parameters, and response parsing. Features automatic retries, timeout handling, SSL verification control, and detailed error reporting. Ideal for microservice communication, third-party API integration, and webhook handling.

## Required Capabilities

Check all capabilities this skill requires:

- [ ] File system access (read)
- [ ] File system access (write)
- [x] Code execution (bash)
- [x] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [ ] User interaction (questions)

## Dependencies

**Required Skills**: None
**External Dependencies**:
- `aiohttp` (recommended for async) or `requests` (synchronous)
- Optional: `httpx` (HTTP/2 support)
- Optional: `certifi` (SSL certificate validation)

**HoloLoom Integration**: Integrates with data pipelines, webhook handlers, microservice communication, and external API workflows.

## Input Schema

```json
{
  "operation": "string - get|post|put|patch|delete|head",
  "parameters": {
    "url": "string (required) - Target URL",
    "headers": "object (optional) - Custom HTTP headers",
    "params": "object (optional) - Query string parameters",
    "json": "object (optional) - JSON request body",
    "data": "object (optional) - Form data request body",
    "files": "object (optional) - Multipart file uploads",
    "auth": {
      "type": "string - bearer|basic|api_key",
      "token": "string - Auth token/API key",
      "username": "string - Username (for basic auth)",
      "password": "string - Password (for basic auth)"
    },
    "timeout": "number (optional) - Request timeout in seconds (default: 30)",
    "max_retries": "number (optional) - Max retry attempts (default: 3)",
    "verify_ssl": "boolean (optional) - Verify SSL certificates (default: true)",
    "follow_redirects": "boolean (optional) - Follow HTTP redirects (default: true)"
  }
}
```

## Output Schema

```json
{
  "status": "string - success|failure|error",
  "result": "object - HTTP response",
  "message": "string - Human-readable summary",
  "execution_time_ms": "number - Skill execution time",
  "details": {
    "operation": "string - HTTP method used",
    "url": "string - Request URL",
    "status_code": "number - HTTP status code (200, 404, etc.)",
    "headers": "object - Response headers",
    "body": "object|string - Response body (parsed JSON or raw text)",
    "latency_ms": "number - Network latency",
    "size_bytes": "number - Response size",
    "redirects": "number - Number of redirects followed",
    "retry_count": "number - Number of retries performed"
  },
  "warnings": "array - Any warnings",
  "errors": "array - Execution errors"
}
```

## Examples

### Example 1: GET Request with Query Parameters

**Input**:
```json
{
  "operation": "get",
  "parameters": {
    "url": "https://api.example.com/users",
    "params": {
      "page": 2,
      "per_page": 50,
      "sort": "created_at"
    },
    "headers": {
      "Accept": "application/json",
      "User-Agent": "HoloLoom/1.0"
    },
    "timeout": 10
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "method": "GET",
    "url": "https://api.example.com/users?page=2&per_page=50&sort=created_at",
    "status_code": 200,
    "headers": {
      "content-type": "application/json; charset=utf-8",
      "x-total-count": "1523"
    },
    "body": {
      "users": [
        {"id": 51, "name": "Alice", "email": "alice@example.com"},
        {"id": 52, "name": "Bob", "email": "bob@example.com"}
      ],
      "page": 2,
      "total_pages": 31
    },
    "latency_ms": 180
  },
  "message": "GET request successful (200 OK)",
  "execution_time_ms": 195
}
```

**Explanation**: Fetches paginated user list with query parameters. Demonstrates URL construction, custom headers, and JSON response parsing.

### Example 2: POST Request with JSON Body

**Input**:
```json
{
  "operation": "post",
  "parameters": {
    "url": "https://api.example.com/users",
    "headers": {
      "Content-Type": "application/json",
      "Authorization": "Bearer eyJhbGciOiJIUzI1NiIs..."
    },
    "json": {
      "name": "Charlie",
      "email": "charlie@example.com",
      "role": "admin"
    },
    "timeout": 15
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "method": "POST",
    "url": "https://api.example.com/users",
    "status_code": 201,
    "headers": {
      "content-type": "application/json",
      "location": "/users/153"
    },
    "body": {
      "id": 153,
      "name": "Charlie",
      "email": "charlie@example.com",
      "role": "admin",
      "created_at": "2025-11-24T10:30:00Z"
    },
    "latency_ms": 250
  },
  "message": "POST request successful (201 Created)",
  "execution_time_ms": 265
}
```

**Explanation**: Creates a new user resource with JSON payload. Shows Bearer token authentication and 201 Created response handling.

### Example 3: PUT Request with Retry Logic

**Input**:
```json
{
  "operation": "put",
  "parameters": {
    "url": "https://api.example.com/users/153",
    "auth": {
      "type": "basic",
      "username": "admin",
      "password": "secret123"
    },
    "json": {
      "name": "Charlie Smith",
      "email": "charlie.smith@example.com"
    },
    "max_retries": 3,
    "timeout": 20
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "method": "PUT",
    "url": "https://api.example.com/users/153",
    "status_code": 200,
    "headers": {
      "content-type": "application/json"
    },
    "body": {
      "id": 153,
      "name": "Charlie Smith",
      "email": "charlie.smith@example.com",
      "updated_at": "2025-11-24T10:35:00Z"
    },
    "latency_ms": 320,
    "retry_count": 1
  },
  "message": "PUT request successful (200 OK) after 1 retry",
  "execution_time_ms": 850
}
```

**Explanation**: Updates existing user with automatic retry on network failure. Demonstrates Basic authentication and retry logic.

### Example 4: DELETE Request

**Input**:
```json
{
  "operation": "delete",
  "parameters": {
    "url": "https://api.example.com/users/153",
    "headers": {
      "Authorization": "Bearer eyJhbGciOiJIUzI1NiIs..."
    },
    "timeout": 10
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "method": "DELETE",
    "url": "https://api.example.com/users/153",
    "status_code": 204,
    "headers": {},
    "body": null,
    "latency_ms": 150
  },
  "message": "DELETE request successful (204 No Content)",
  "execution_time_ms": 165
}
```

**Explanation**: Deletes a resource with proper authentication. Shows 204 No Content response handling (empty body).

### Example 5: PATCH Request with Partial Update

**Input**:
```json
{
  "operation": "patch",
  "parameters": {
    "url": "https://api.example.com/users/153",
    "headers": {
      "Content-Type": "application/json",
      "Authorization": "API-Key abc123xyz789"
    },
    "json": {
      "role": "moderator"
    },
    "timeout": 15
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "method": "PATCH",
    "url": "https://api.example.com/users/153",
    "status_code": 200,
    "headers": {
      "content-type": "application/json"
    },
    "body": {
      "id": 153,
      "name": "Charlie Smith",
      "email": "charlie.smith@example.com",
      "role": "moderator",
      "updated_at": "2025-11-24T10:40:00Z"
    },
    "latency_ms": 180
  },
  "message": "PATCH request successful (200 OK)",
  "execution_time_ms": 195
}
```

**Explanation**: Partially updates a resource (only role field). Demonstrates API key authentication and PATCH semantics.

## Testing Checklist

- [x] **Functionality**: All 6 HTTP methods execute correctly
- [x] **Error Handling**: Graceful handling of network errors, timeouts, 4xx/5xx responses
- [x] **Security**: No credential logging, safe URL construction
- [x] **Performance**: Operations complete within expected time (<5s)
- [x] **Token Efficiency**: Structured output, minimal verbosity
- [x] **Documentation**: All sections complete
- [x] **Dependencies**: HTTP library documented
- [x] **Edge Cases**: Handles redirects, retries, SSL errors, large responses
- [x] **Output Consistency**: Consistent result structure
- [x] **Integration**: Works with HoloLoom API integration pipelines

## Security Considerations

**Potential Risks**:
- **Credential Exposure**: Auth tokens in logs -> Never log credentials or tokens
- **Server-Side Request Forgery (SSRF)**: Unrestricted URL access -> Validate and whitelist URLs
- **SSL Verification Bypass**: Disabling SSL -> Only allow in dev/test environments

**Data Privacy**:
- [x] Does not log sensitive request/response bodies
- [x] Does not cache credentials
- [x] Does not send data to unauthorized endpoints

**Sandboxing**:
- [x] Operates within defined capability boundaries
- [x] URL validation prevents SSRF attacks
- [x] Timeouts prevent indefinite hangs

## Performance Characteristics

- **Expected Latency**: 100-5000ms (1-5 seconds depending on network and API)
- **Token Usage**: 100-1000 tokens per execution
- **Resource Requirements**: Network connectivity, DNS resolution
- **Scalability**: Limited by network bandwidth and API rate limits

**Operation-Specific Latencies**:
- `get`: 100-2000ms (depends on response size)
- `post`: 200-3000ms (includes request upload time)
- `put`: 200-3000ms (includes request upload time)
- `patch`: 150-2000ms (smaller payloads than PUT)
- `delete`: 100-1500ms (typically fast, no response body)
- `head`: 50-500ms (no body transfer, fastest)

## License

MIT License

## Related Documentation

- **aiohttp Docs**: [docs.aiohttp.org](https://docs.aiohttp.org)
- **HTTP Status Codes**: [httpstatuses.com](https://httpstatuses.com)
- **REST API Best Practices**: [restfulapi.net](https://restfulapi.net)
- **HoloLoom Web Skills**: [../README.md](../README.md)
