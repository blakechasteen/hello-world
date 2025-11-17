# Promptly REST API

Production-ready REST API for Promptly - Advanced prompt management with versioning, branching, and evaluation.

## Features

- **Complete CRUD Operations**: Create, read, update, and list prompts
- **Version Control**: Branch management, commit history, and diff capabilities
- **Evaluation System**: Run tests, compare results, and track performance
- **Chain Execution**: Sequential prompt execution with state management
- **Plugin System**: Extensible evaluators and storage backends
- **WebSocket Support**: Real-time updates for evaluations and chains
- **Production Ready**: Authentication, rate limiting, logging, and monitoring
- **Python SDK**: Both synchronous and asynchronous client libraries
- **Docker Deployment**: Complete containerized setup with docker-compose

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Initialize Promptly repository
cd /path/to/your/prompts
python -c "from promptly import Promptly; Promptly().init()"

# Run the API server
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000/docs for interactive API documentation.

### Docker Deployment

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## API Endpoints

### System

- `GET /health` - Health check
- `GET /` - API information
- `GET /docs` - Interactive API documentation
- `GET /redoc` - ReDoc documentation

### Authentication

- `POST /api/v1/auth/token` - Get JWT token
- `POST /api/v1/auth/api-keys` - Create API key
- `GET /api/v1/auth/api-keys` - List API keys
- `DELETE /api/v1/auth/api-keys/{key_id}` - Delete API key
- `GET /api/v1/auth/me` - Get current user

### Prompts

- `POST /api/v1/prompts` - Create/update prompt
- `GET /api/v1/prompts` - List all prompts
- `GET /api/v1/prompts/{name}` - Get specific prompt
- `POST /api/v1/prompts/search` - Search prompts
- `GET /api/v1/prompts/{name}/diff` - Get diff between versions

### Branches

- `POST /api/v1/branches` - Create branch
- `GET /api/v1/branches` - List all branches
- `GET /api/v1/branches/{name}` - Get branch details
- `POST /api/v1/branches/checkout` - Checkout branch
- `DELETE /api/v1/branches/{name}` - Delete branch

### History

- `GET /api/v1/history/log` - Get commit history
- `GET /api/v1/history/blame/{name}` - Get blame information

### Evaluations

- `POST /api/v1/evaluations` - Run evaluation
- `GET /api/v1/evaluations` - List evaluations
- `GET /api/v1/evaluations/{id}` - Get evaluation results
- `POST /api/v1/evaluations/compare` - Compare evaluations

### Chains

- `POST /api/v1/chains` - Create chain
- `GET /api/v1/chains` - List chains
- `GET /api/v1/chains/{name}` - Get chain details
- `POST /api/v1/chains/execute` - Execute chain
- `GET /api/v1/chains/executions/{id}` - Get execution status
- `DELETE /api/v1/chains/{name}` - Delete chain

### Plugins

- `GET /api/v1/plugins` - List all plugins
- `GET /api/v1/plugins/{type}/{name}` - Get plugin details
- `POST /api/v1/plugins/configure` - Configure plugin

### WebSocket

- `WS /ws/{client_id}` - Main WebSocket connection
- `WS /ws/evaluations/{evaluation_id}` - Stream evaluation
- `WS /ws/chains/{execution_id}` - Stream chain execution
- `GET /ws/stats` - WebSocket statistics

## Authentication

The API supports two authentication methods:

### API Key Authentication

Include the API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/prompts
```

Development API Key: `pk_dev_key_12345`

### JWT Authentication

1. Get a token:

```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin&password=admin"
```

2. Use the token:

```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/prompts
```

## Python SDK

### Synchronous Client

```python
from promptly.sdk import PromptlyClient

# Initialize client
client = PromptlyClient(
    base_url="http://localhost:8000",
    api_key="pk_dev_key_12345"
)

# Create a prompt
client.create_prompt(
    name="summarizer",
    content="Summarize the following text:\n\n{text}",
    metadata={"tags": ["nlp", "summarization"]}
)

# Get a prompt
prompt = client.get_prompt("summarizer")
print(prompt)

# List all prompts
prompts = client.list_prompts()

# Search prompts
results = client.search_prompts(query="summary", tags=["nlp"])

# Create a branch
client.create_branch("feature/new-prompts", from_branch="main")

# Run evaluation
eval_result = client.run_evaluation(
    prompt_name="summarizer",
    test_cases=[
        {
            "inputs": {"text": "Long text here..."},
            "expected": "Expected summary..."
        }
    ],
    evaluator="semantic"
)

# Create and execute chain
client.create_chain(
    name="research-pipeline",
    steps=["query-expander", "searcher", "summarizer"]
)

result = client.execute_chain(
    chain_name="research-pipeline",
    initial_input={"query": "AI trends"}
)
```

### Asynchronous Client

```python
import asyncio
from promptly.sdk import AsyncPromptlyClient

async def main():
    async with AsyncPromptlyClient(
        base_url="http://localhost:8000",
        api_key="pk_dev_key_12345"
    ) as client:
        # Create prompt
        await client.create_prompt("test", "Content: {input}")

        # Get prompt
        prompt = await client.get_prompt("test")

        # Run evaluation
        result = await client.run_evaluation(
            prompt_name="test",
            test_cases=[{"inputs": {"input": "test"}}]
        )

asyncio.run(main())
```

## WebSocket Usage

```python
import asyncio
import websockets
import json

async def stream_evaluation():
    uri = "ws://localhost:8000/ws/evaluations/eval_123?api_key=pk_dev_key_12345"

    async with websockets.connect(uri) as websocket:
        # Listen for updates
        async for message in websocket:
            data = json.loads(message)

            if data["type"] == "evaluation_progress":
                print(f"Progress: {data['progress']}%")

            elif data["type"] == "evaluation_result":
                print(f"Result: {data['result']}")

            elif data["type"] == "evaluation_complete":
                print("Complete!")
                break

asyncio.run(stream_evaluation())
```

## Configuration

Environment variables (see `.env.example`):

```bash
# API Configuration
PROMPTLY_HOST=0.0.0.0
PROMPTLY_PORT=8000
PROMPTLY_WORKERS=4

# Security
PROMPTLY_SECRET_KEY=your-secret-key
PROMPTLY_API_KEY_HEADER=X-API-Key

# Rate Limiting
PROMPTLY_RATE_LIMIT_ENABLED=true
PROMPTLY_RATE_LIMIT_PER_MINUTE=60

# CORS
PROMPTLY_CORS_ORIGINS=["http://localhost:3000"]

# Storage
PROMPTLY_ROOT_DIR=/path/to/prompts
PROMPTLY_STORAGE_BACKEND=sqlite
```

## Testing

```bash
# Run all tests
pytest api/tests/ -v

# Run specific test file
pytest api/tests/test_api.py -v

# Run with coverage
pytest api/tests/ --cov=api --cov-report=html
```

## Monitoring

### Health Checks

```bash
curl http://localhost:8000/health
```

### Metrics

Prometheus metrics available at `:9090/metrics` (if enabled)

### Logs

Structured JSON logging to stdout:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "message": "Request completed",
  "request_id": "abc123",
  "method": "GET",
  "path": "/api/v1/prompts",
  "status_code": 200,
  "duration_ms": 45.2
}
```

## Production Deployment

### Security Checklist

- [ ] Change `PROMPTLY_SECRET_KEY` to a random value
- [ ] Use environment variables for sensitive data
- [ ] Enable HTTPS with valid SSL certificate
- [ ] Configure CORS for your domain
- [ ] Set up proper firewall rules
- [ ] Use strong API keys
- [ ] Enable rate limiting
- [ ] Set up monitoring and alerting
- [ ] Regular security updates

### Docker Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    image: promptly-api:latest
    environment:
      - PROMPTLY_SECRET_KEY=${SECRET_KEY}
      - PROMPTLY_WORKERS=8
      - PROMPTLY_LOG_LEVEL=WARNING
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
```

### Nginx Configuration

See `nginx.conf` for reverse proxy setup with:
- SSL/TLS termination
- Rate limiting
- Security headers
- WebSocket support

## Troubleshooting

### Common Issues

**API not starting:**
- Check if port 8000 is available
- Verify Python version (3.11+)
- Ensure all dependencies are installed

**Authentication errors:**
- Verify API key in `X-API-Key` header
- Check if key is valid
- Ensure Promptly repository is initialized

**Rate limiting:**
- Wait for rate limit to reset
- Adjust `PROMPTLY_RATE_LIMIT_PER_MINUTE`
- Use different API keys for different clients

## Support

- Documentation: http://localhost:8000/docs
- Issues: GitHub Issues
- Email: support@promptly.example

## License

MIT License - see LICENSE file for details
