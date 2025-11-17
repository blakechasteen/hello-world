#!/usr/bin/env python3
"""
Configuration Generator
======================

Generates platform-specific configuration files.

Supports:
- Heroku (Procfile, app.json, heroku.yml)
- Railway (railway.toml, .railway.json)
- Fly.io (fly.toml)
- AWS Lambda (serverless.yml, lambda_function.py)
- Local Docker (docker-compose.yml, .env)

Created: November 2025
"""

from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class ConfigGenerator:
    """
    Generate platform-specific configuration files.

    Each platform requires different config files for deployment.
    This class generates them all automatically.
    """

    @staticmethod
    def generate_heroku_config(workflow_id: str, dest_dir: Path) -> None:
        """Generate Heroku configuration files."""
        # Procfile
        (dest_dir / 'Procfile').write_text(
            "web: uvicorn main:app --host 0.0.0.0 --port $PORT\n"
        )

        # app.json
        app_json = {
            "name": f"hololoom-{workflow_id}",
            "description": f"HoloLoom workflow: {workflow_id}",
            "repository": "https://github.com/your-org/hololoom",
            "keywords": ["hololoom", "workflow", "automation"],
            "addons": ["heroku-postgresql:hobby-basic"],
            "env": {
                "WORKFLOW_ID": {"value": workflow_id},
                "HOLOLOOM_ENV": {"value": "production"},
                "LOG_LEVEL": {"value": "INFO"}
            },
            "formation": {
                "web": {"quantity": 1, "size": "hobby"}
            }
        }

        import json
        (dest_dir / 'app.json').write_text(json.dumps(app_json, indent=2))

        # heroku.yml (for Docker deploys)
        (dest_dir / 'heroku.yml').write_text("""build:
  docker:
    web: Dockerfile
run:
  web: uvicorn main:app --host 0.0.0.0 --port $PORT
""")

    @staticmethod
    def generate_railway_config(workflow_id: str, dest_dir: Path) -> None:
        """Generate Railway configuration files."""
        # railway.toml
        (dest_dir / 'railway.toml').write_text(f"""[build]
builder = "NIXPACKS"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port $PORT"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10

[env]
WORKFLOW_ID = "{workflow_id}"
HOLOLOOM_ENV = "production"
LOG_LEVEL = "INFO"
""")

    @staticmethod
    def generate_fly_config(workflow_id: str, dest_dir: Path) -> None:
        """Generate Fly.io configuration files."""
        # fly.toml
        (dest_dir / 'fly.toml').write_text(f"""app = "hololoom-{workflow_id}"
primary_region = "iad"

[build]

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0
  processes = ["app"]

[[services]]
  internal_port = 8000
  protocol = "tcp"

  [[services.ports]]
    handlers = ["http"]
    port = 80

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 256
""")

    @staticmethod
    def generate_aws_lambda_config(workflow_id: str, dest_dir: Path) -> None:
        """Generate AWS Lambda configuration files."""
        # serverless.yml
        (dest_dir / 'serverless.yml').write_text(f"""service: hololoom-{workflow_id}

provider:
  name: aws
  runtime: python3.11
  region: us-east-1
  memorySize: 512
  timeout: 30

functions:
  app:
    handler: lambda_function.handler
    events:
      - httpApi: '*'

plugins:
  - serverless-python-requirements
""")

        # lambda_function.py
        (dest_dir / 'lambda_function.py').write_text("""from mangum import Mangum
from main import app

handler = Mangum(app, lifespan="off")
""")

    @staticmethod
    def generate_docker_config(workflow_id: str, dest_dir: Path) -> None:
        """Generate Local Docker configuration files."""
        # docker-compose.yml
        (dest_dir / 'docker-compose.yml').write_text(f"""version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - WORKFLOW_ID={workflow_id}
      - HOLOLOOM_ENV=local
      - DATABASE_URL=postgresql://postgres:password@db:5432/hololoom
    depends_on:
      - db
    volumes:
      - ./:/app

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=hololoom
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
""")
