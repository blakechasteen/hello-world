# Skill: Dockerfile Generator

## Metadata

- **Name**: `dockerfile_generator`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-22`
- **Last Updated**: `2025-11-22`
- **Category**: `domain`
- **Tags**: `docker, devops, containerization, deployment`

## Description

**Short Description**:
Generates production-ready Dockerfiles and docker-compose.yml files from project structure analysis with best practices and multi-stage builds.

**Detailed Description**:
Creating optimal Dockerfiles requires deep knowledge of Docker best practices, security, and language-specific optimization. This skill analyzes project structure (package.json, requirements.txt, go.mod, etc.), detects language/framework, generates production-ready Dockerfiles with multi-stage builds, security hardening, layer caching optimization, and companion docker-compose.yml for local development. Supports Node.js, Python, Go, Java, Rust, and static sites.

## Required Capabilities

- [x] File system access (read)
- [ ] File system access (write)
- [ ] Code execution (bash)
- [ ] Code execution (python)
- [ ] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [ ] User interaction (questions)

## Dependencies

**Required Skills**: None
**External Dependencies**: None
**HoloLoom Integration**: None

## Input Schema

```json
{
  "project_structure": {
    "language": "string - nodejs|python|go|java|rust|static",
    "files": ["array of filenames"],
    "framework": "string (optional) - express|flask|django|gin|spring|etc",
    "build_tool": "string (optional) - npm|yarn|pnpm|pip|go|maven|cargo"
  },
  "requirements": [
    "string - Additional requirements (e.g., 'include redis', 'optimize for size')"
  ]
}
```

## Output Schema

```json
{
  "dockerfile": "string - Production Dockerfile content",
  "docker_compose": "string - docker-compose.yml for local development",
  "dockerignore": "string - .dockerignore content",
  "best_practices": [
    "string - List of best practices applied"
  ],
  "build_instructions": "string - How to build and run",
  "metadata": {
    "estimated_image_size": "string - e.g., '100MB'",
    "build_stages": "number",
    "security_hardening": ["array of security measures"],
    "confidence": "number (0.0-1.0)"
  }
}
```

## Prompt Template

```markdown
You are a DevOps expert creating production-ready Dockerfiles.

**Project Structure**:
{project_structure}

**Additional Requirements**:
{requirements}

**Your Task**:
1. Detect language, framework, and dependencies
2. Generate multi-stage Dockerfile (builder + runtime)
3. Create docker-compose.yml for local development
4. Generate .dockerignore for efficient builds
5. List best practices applied
6. Provide build/run instructions

**Docker Best Practices**:
- Use official base images (node:alpine, python:slim, etc.)
- Multi-stage builds (separate build and runtime)
- Run as non-root user
- Minimize layers (combine RUN commands)
- Optimize layer caching (COPY dependencies first)
- Use .dockerignore to exclude unnecessary files
- Pin versions (node:20-alpine, not node:alpine)
- Health checks for production
- Security scanning (no secrets in layers)

**Language-Specific Optimizations**:
- Node.js: npm ci (not npm install), remove devDependencies
- Python: pip install --no-cache-dir, use wheels
- Go: Static binary compilation, scratch/distroless base
- Java: Use JRE not JDK, multi-stage with Maven/Gradle
- Rust: cargo build --release, musl target for Alpine

Return structured JSON matching the output schema.
```

## Examples

### Example 1: Node.js Express App

**Input**:
```json
{
  "project_structure": {
    "language": "nodejs",
    "files": ["package.json", "package-lock.json", "server.js", "src/"],
    "framework": "express",
    "build_tool": "npm"
  },
  "requirements": ["optimize for size"]
}
```

**Expected Output**:
```json
{
  "dockerfile": "# Multi-stage build for Node.js Express app\n\n# Stage 1: Dependencies\nFROM node:20-alpine AS deps\nWORKDIR /app\nCOPY package.json package-lock.json ./\nRUN npm ci --only=production\n\n# Stage 2: Build\nFROM node:20-alpine AS build\nWORKDIR /app\nCOPY package.json package-lock.json ./\nRUN npm ci\nCOPY . .\nRUN npm run build\n\n# Stage 3: Production\nFROM node:20-alpine AS production\n\n# Security: Run as non-root user\nRUN addgroup -g 1001 -S nodejs && adduser -S nodejs -u 1001\n\nWORKDIR /app\n\n# Copy only production dependencies and built app\nCOPY --from=deps --chown=nodejs:nodejs /app/node_modules ./node_modules\nCOPY --from=build --chown=nodejs:nodejs /app/dist ./dist\nCOPY --chown=nodejs:nodejs package.json ./\n\nUSER nodejs\n\nEXPOSE 3000\n\nHEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\\n  CMD node -e \"require('http').get('http://localhost:3000/health', (r) => process.exit(r.statusCode === 200 ? 0 : 1))\"\n\nCMD [\"node\", \"dist/server.js\"]",
  "docker_compose": "version: '3.8'\n\nservices:\n  app:\n    build:\n      context: .\n      dockerfile: Dockerfile\n    ports:\n      - '3000:3000'\n    environment:\n      - NODE_ENV=development\n    volumes:\n      - .:/app\n      - /app/node_modules\n    command: npm run dev\n\n  # Optional: Add database, redis, etc.\n  # db:\n  #   image: postgres:15-alpine\n  #   environment:\n  #     POSTGRES_PASSWORD: postgres\n  #   volumes:\n  #     - postgres_data:/var/lib/postgresql/data\n\n# volumes:\n#   postgres_data:",
  "dockerignore": "node_modules\nnpm-debug.log\n.git\n.gitignore\n.env\n.env.local\nREADME.md\n.vscode\n.idea\n*.log\ndist\nbuild\ncoverage\n.DS_Store",
  "best_practices": [
    "Multi-stage build (3 stages: deps, build, production)",
    "Alpine Linux base for minimal image size (~50MB)",
    "npm ci for reproducible builds",
    "Production-only dependencies in final image",
    "Non-root user (nodejs:nodejs)",
    "Health check endpoint",
    "Layer caching optimization (COPY package files first)",
    ".dockerignore to exclude unnecessary files"
  ],
  "build_instructions": "# Build production image\ndocker build -t myapp:latest .\n\n# Run production container\ndocker run -p 3000:3000 myapp:latest\n\n# Or use docker-compose for local development\ndocker-compose up",
  "metadata": {
    "estimated_image_size": "50-80MB (Alpine base + Node.js + app)",
    "build_stages": 3,
    "security_hardening": [
      "Non-root user",
      "No secrets in layers",
      "Pinned base image version",
      "Health check for monitoring"
    ],
    "confidence": 0.95
  }
}
```

### Example 2: Python Flask App

**Input**:
```json
{
  "project_structure": {
    "language": "python",
    "files": ["requirements.txt", "app.py", "wsgi.py"],
    "framework": "flask",
    "build_tool": "pip"
  },
  "requirements": ["include gunicorn"]
}
```

**Expected Output**:
```json
{
  "dockerfile": "# Multi-stage build for Python Flask app\n\n# Stage 1: Dependencies\nFROM python:3.11-slim AS deps\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt gunicorn\n\n# Stage 2: Production\nFROM python:3.11-slim AS production\n\n# Security: Run as non-root user\nRUN useradd -m -u 1001 appuser\n\nWORKDIR /app\n\n# Copy dependencies and app\nCOPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages\nCOPY --from=deps /usr/local/bin /usr/local/bin\nCOPY --chown=appuser:appuser . .\n\nUSER appuser\n\nEXPOSE 8000\n\nHEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\\n  CMD python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\"\n\nCMD [\"gunicorn\", \"--bind\", \"0.0.0.0:8000\", \"--workers\", \"4\", \"wsgi:app\"]",
  "docker_compose": "version: '3.8'\n\nservices:\n  app:\n    build:\n      context: .\n      dockerfile: Dockerfile\n    ports:\n      - '8000:8000'\n    environment:\n      - FLASK_ENV=development\n    volumes:\n      - .:/app\n    command: flask run --host=0.0.0.0 --port=8000",
  "dockerignore": "__pycache__\n*.pyc\n*.pyo\n*.pyd\n.Python\n.env\n.venv\nvenv/\nENV/\n.git\n.gitignore\nREADME.md\n.pytest_cache\n.coverage\nhtmlcov/\n*.log",
  "best_practices": [
    "Multi-stage build",
    "Python slim base (~150MB vs ~900MB full)",
    "pip --no-cache-dir to reduce image size",
    "Non-root user (appuser)",
    "Gunicorn for production (not Flask dev server)",
    "Health check endpoint",
    "4 workers for parallelism"
  ],
  "build_instructions": "# Build\ndocker build -t flask-app:latest .\n\n# Run\ndocker run -p 8000:8000 flask-app:latest\n\n# Or docker-compose for dev\ndocker-compose up",
  "metadata": {
    "estimated_image_size": "150-200MB",
    "build_stages": 2,
    "security_hardening": [
      "Non-root user",
      "Slim base image",
      "No cache in pip install"
    ],
    "confidence": 0.92
  }
}
```

### Example 3: Go API

**Input**:
```json
{
  "project_structure": {
    "language": "go",
    "files": ["go.mod", "go.sum", "main.go", "cmd/", "pkg/"],
    "framework": "gin",
    "build_tool": "go"
  },
  "requirements": ["minimize image size"]
}
```

**Expected Output**:
```json
{
  "dockerfile": "# Multi-stage build for Go app\n\n# Stage 1: Build\nFROM golang:1.21-alpine AS build\n\nWORKDIR /app\n\n# Copy go mod files\nCOPY go.mod go.sum ./\nRUN go mod download\n\n# Copy source\nCOPY . .\n\n# Build static binary\nRUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags='-w -s' -o /app/server ./cmd\n\n# Stage 2: Production (scratch for minimal size)\nFROM scratch\n\n# Copy SSL certificates for HTTPS\nCOPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/\n\n# Copy binary\nCOPY --from=build /app/server /server\n\nEXPOSE 8080\n\nCMD [\"/server\"]",
  "docker_compose": "version: '3.8'\n\nservices:\n  app:\n    build:\n      context: .\n      dockerfile: Dockerfile\n    ports:\n      - '8080:8080'\n    environment:\n      - GIN_MODE=release",
  "dockerignore": "*.exe\n*.test\n.git\n.gitignore\nREADME.md\n.vscode\n.idea\n*.log",
  "best_practices": [
    "Multi-stage build (builder + scratch)",
    "Scratch base image (~5MB total!)",
    "Static binary compilation (CGO_ENABLED=0)",
    "Stripped binary (-ldflags='-w -s')",
    "SSL certs included for HTTPS",
    "Layer caching (go mod download before source copy)"
  ],
  "build_instructions": "# Build\ndocker build -t go-api:latest .\n\n# Run\ndocker run -p 8080:8080 go-api:latest",
  "metadata": {
    "estimated_image_size": "5-10MB (scratch + static binary)",
    "build_stages": 2,
    "security_hardening": [
      "Minimal attack surface (scratch base)",
      "Static binary (no dependencies)",
      "SSL certificates included"
    ],
    "confidence": 0.98
  }
}
```

## Testing Checklist

- [x] **Functionality**: Generates valid Dockerfiles
- [x] **Error Handling**: Handles unknown languages gracefully
- [x] **Security**: No secret exposure
- [x] **Performance**: < 1s per generation
- [x] **Token Efficiency**: ~600 tokens
- [x] **Documentation**: Complete
- [x] **Dependencies**: File read only
- [x] **Edge Cases**: Multiple languages, custom requirements
- [x] **Output Consistency**: Structured JSON
- [x] **Integration**: Standalone

## Security Considerations

**Potential Risks**:
- Generated Dockerfiles might have vulnerabilities
  - **Mitigation**: Follow security best practices, recommend scanning
**Data Privacy**:
- [x] Does not access external resources
**Sandboxing**:
- [x] File read only (no execution)

## Performance Characteristics

- **Expected Latency**: 500ms - 1s
- **Token Usage**: ~600 tokens
- **Resource Requirements**: Minimal
- **Scalability**: O(1) per project

## Maintenance Notes

**Known Limitations**:
- Generic templates (not project-specific optimization)
- Covers 6 main languages

**Future Enhancements**:
- Kubernetes manifests generation
- Docker security scanning integration
- Custom base image recommendations

**Changelog**:
- **v1.0.0** (2025-11-22): Initial release

## License

MIT License
