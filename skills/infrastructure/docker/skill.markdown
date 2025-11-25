# Skill: Docker

## Metadata

- **Name**: `docker`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-24`
- **Last Updated**: `2025-11-24`
- **Category**: `infrastructure`
- **Tags**: `docker, containers, devops, deployment, orchestration`

## Description

**Short Description**:
Container orchestration and lifecycle management with Docker and Docker Compose.

**Detailed Description**:
The Docker skill provides comprehensive container and image management capabilities including building images, running containers, orchestrating multi-container applications with Docker Compose, inspecting resources, managing logs, and cleanup operations. Wraps Docker CLI and Docker Compose for seamless integration with containerized applications. Enables automated deployment, testing in isolated environments, and infrastructure-as-code workflows directly from HoloLoom.

## Required Capabilities

Check all capabilities this skill requires:

- [x] File system access (read)
- [ ] File system access (write)
- [x] Code execution (bash)
- [ ] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [ ] User interaction (questions)

## Dependencies

**Required Skills**: None
**External Dependencies**:
- `docker` CLI (required)
- `docker-compose` or `docker compose` (optional, for compose operations)
- Docker daemon running
- User permissions for Docker socket

**HoloLoom Integration**: Integrates with HoloLoom's deployment pipeline, testing infrastructure (containerized tests), and development environment setup.

## Input Schema

```json
{
  "operation": "string - ps|build|run|stop|logs|compose_up|compose_down|inspect|prune",
  "parameters": {
    "image": "string (required for build/run) - Image name or Dockerfile path",
    "container": "string (required for stop/logs/inspect) - Container ID or name",
    "tag": "string (optional for build) - Image tag",
    "ports": "object (optional for run) - Port mappings",
    "env": "object (optional for run) - Environment variables",
    "volumes": "object (optional for run) - Volume mounts",
    "compose_file": "string (optional for compose, default: docker-compose.yml) - Compose file path",
    "service": "string (optional for compose) - Specific service to operate on",
    "follow": "boolean (optional for logs, default: false) - Follow log output",
    "tail": "number (optional for logs) - Number of recent log lines"
  }
}
```

## Output Schema

```json
{
  "status": "string - success|failure|error",
  "result": "object|array - Operation-specific result",
  "message": "string - Human-readable summary",
  "execution_time_ms": "number - Skill execution time",
  "details": {
    "operation": "string - Operation performed",
    "containers": "array (for ps) - Running containers",
    "image_id": "string (for build) - Built image ID",
    "container_id": "string (for run) - Started container ID",
    "logs": "string (for logs) - Container logs",
    "inspect_data": "object (for inspect) - Container/image metadata"
  },
  "warnings": "array - Any warnings",
  "errors": "array - Execution errors"
}
```

## Examples

### Example 1: Build Docker Image

**Input**:
```json
{
  "operation": "build",
  "parameters": {
    "image": ".",
    "tag": "myapp:latest"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "image_id": "sha256:abc123...",
    "tag": "myapp:latest",
    "size_mb": 245.6
  },
  "message": "Docker image 'myapp:latest' built successfully",
  "execution_time_ms": 15234
}
```

### Example 2: Run Container

**Input**:
```json
{
  "operation": "run",
  "parameters": {
    "image": "myapp:latest",
    "ports": {"8000": "8000"},
    "env": {"DATABASE_URL": "postgres://..."}
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "container_id": "abc123def456",
    "image": "myapp:latest",
    "ports": {"8000/tcp": "8000"},
    "status": "running"
  },
  "message": "Container started successfully",
  "execution_time_ms": 1250
}
```

### Example 3: Docker Compose Up

**Input**:
```json
{
  "operation": "compose_up",
  "parameters": {
    "compose_file": "docker-compose.yml"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "services_started": ["web", "db", "redis"],
    "containers": [
      {"name": "myapp_web_1", "status": "running"},
      {"name": "myapp_db_1", "status": "running"},
      {"name": "myapp_redis_1", "status": "running"}
    ]
  },
  "message": "Docker Compose stack started (3 services)",
  "execution_time_ms": 5670
}
```

### Example 4: Get Container Logs

**Input**:
```json
{
  "operation": "logs",
  "parameters": {
    "container": "myapp_web_1",
    "tail": 100
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "container": "myapp_web_1",
    "logs": "2025-11-24 10:00:00 Starting application...\n2025-11-24 10:00:01 Server listening on port 8000\n...",
    "log_lines": 100
  },
  "message": "Retrieved 100 log lines",
  "execution_time_ms": 450
}
```

### Example 5: List Running Containers

**Input**:
```json
{
  "operation": "ps",
  "parameters": {}
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "containers": [
      {
        "id": "abc123",
        "image": "myapp:latest",
        "command": "python app.py",
        "created": "2025-11-24T10:00:00Z",
        "status": "Up 30 minutes",
        "ports": "0.0.0.0:8000->8000/tcp",
        "names": "myapp_web_1"
      }
    ],
    "total": 1
  },
  "message": "Found 1 running container",
  "execution_time_ms": 320
}
```

## Testing Checklist

- [x] **Functionality**: All 9 operations execute correctly
- [x] **Error Handling**: Graceful handling of Docker daemon errors
- [x] **Security**: No command injection, secure env var handling
- [x] **Performance**: Operations complete within expected time
- [x] **Token Efficiency**: Structured output, minimal verbosity
- [x] **Documentation**: All sections complete
- [x] **Dependencies**: Docker CLI documented
- [x] **Edge Cases**: Handles missing images, stopped containers
- [x] **Output Consistency**: Consistent result structure
- [x] **Integration**: Works with HoloLoom deployment pipeline

## Security Considerations

**Potential Risks**:
- **Command Injection**: Container names could contain shell commands → Sanitize inputs
- **Docker Socket Access**: Skill requires Docker socket access → Validate permissions
- **Resource Exhaustion**: Running many containers → Implement resource limits

**Data Privacy**:
- [x] Does not log sensitive environment variables
- [x] Does not expose Docker socket outside localhost
- [x] Does not make unauthorized external requests

**Sandboxing**:
- [x] Operates within defined capability boundaries
- [x] Does not attempt privilege escalation
- [x] Does not modify system files outside Docker scope

## Performance Characteristics

- **Expected Latency**: 100-30000ms (0.1-30 seconds depending on operation)
- **Token Usage**: 150-700 tokens per execution
- **Resource Requirements**: Docker daemon, sufficient disk space for images
- **Scalability**: Limited by host resources (CPU, memory, disk)

## License

MIT License

## Related Documentation

- **Docker Documentation**: [docs.docker.com](https://docs.docker.com/)
- **Docker Compose**: [docs.docker.com/compose](https://docs.docker.com/compose/)
- **HoloLoom Deployment**: [CLAUDE.md](../../../CLAUDE.md)
