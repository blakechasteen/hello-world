#!/usr/bin/env python3
"""
Local Docker Deployer
====================

Deploy HoloLoom workflows locally using Docker Compose.

Features:
- Runs on your machine
- No cloud account needed
- Free
- Great for testing and development

Cost: $0 (free)

Created: November 2025
"""

import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class LocalDockerDeployer:
    """
    Deploy workflows locally with Docker Compose.

    Local Docker is the fallback option:
    - No cloud account needed
    - Runs on your machine
    - Free
    - Perfect for testing

    Cost: $0
    """

    def __init__(self, workflow_id: str, package_path: Optional[Path] = None):
        self.workflow_id = workflow_id
        self.package_path = package_path
        self.container_name = f"hololoom-{workflow_id}"
        self.start_time = None

    async def deploy(self) -> Dict[str, Any]:
        """Deploy workflow locally using Docker Compose."""
        self.start_time = datetime.now()

        try:
            # Check Docker
            print("   Checking Docker...")
            await self._check_docker()
            print("   ✓ Docker found")

            # Create docker-compose.yml
            print(f"   Creating Docker Compose configuration...")
            await self._create_docker_compose()
            print(f"   ✓ Configuration created")

            # Start containers
            print("   Starting containers...")
            await self._start_containers()
            print("   ✓ Containers started")

            duration = (datetime.now() - self.start_time).total_seconds()

            return {
                'status': 'success',
                'platform': 'local_docker',
                'container_name': self.container_name,
                'url': 'http://localhost:8000',
                'dashboard_url': 'http://localhost:8000/analytics',
                'duration_seconds': duration,
                'cost_estimate': {
                    'monthly': 0,
                    'breakdown': {}
                }
            }

        except Exception as e:
            raise Exception(f"Docker deployment failed: {e}")

    async def _check_docker(self) -> None:
        """Check if Docker is installed and running."""
        result = subprocess.run(['docker', '--version'], capture_output=True)
        if result.returncode != 0:
            raise Exception("Docker not found. Install: https://docs.docker.com/get-docker/")

        # Check if Docker daemon is running
        result = subprocess.run(['docker', 'ps'], capture_output=True)
        if result.returncode != 0:
            raise Exception("Docker daemon not running. Start Docker Desktop")

    async def _create_docker_compose(self) -> None:
        """Create docker-compose.yml file."""
        docker_compose_content = f"""version: '3.8'

services:
  web:
    container_name: {self.container_name}
    build: .
    ports:
      - "8000:8000"
    environment:
      - WORKFLOW_ID={self.workflow_id}
      - HOLOLOOM_ENV=local
      - DATABASE_URL=postgresql://postgres:password@db:5432/hololoom
    depends_on:
      - db
    volumes:
      - ./:/app
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

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
"""

        # Write to current directory
        Path('docker-compose.yml').write_text(docker_compose_content)

        # Create Dockerfile
        dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        Path('Dockerfile').write_text(dockerfile_content)

    async def _start_containers(self) -> None:
        """Start Docker Compose containers."""
        result = subprocess.run(
            ['docker-compose', 'up', '-d', '--build'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise Exception(f"Failed to start containers: {result.stderr}")

    def estimate_cost(self) -> Dict[str, Any]:
        """Estimate monthly cost."""
        return {
            'monthly': 0,
            'description': 'Free (runs on your machine)',
            'breakdown': {},
            'notes': [
                'No cloud costs',
                'Uses your local compute resources',
                'Perfect for development and testing'
            ]
        }
