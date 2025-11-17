#!/usr/bin/env python3
"""
Workflow Packager
================

Packages workflows into deployable artifacts for different platforms.

Creates:
- Dockerfile
- requirements.txt
- runtime config
- environment template
- health check endpoint

Created: November 2025
"""

import json
import tarfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class WorkflowPackager:
    """
    Package workflows for deployment.

    Creates platform-specific deployment artifacts including:
    - Application code
    - Dependencies
    - Configuration
    - Health checks
    """

    def __init__(self):
        self.repo_root = Path(__file__).parent.parent.parent
        self.workflows_dir = self.repo_root / 'HoloLoom' / 'workflows' / 'templates'

    def package(self, workflow_id: str, custom_config: Dict[str, Any]) -> Path:
        """
        Package workflow into deployable artifact.

        Args:
            workflow_id: ID of the workflow to package
            custom_config: Custom configuration options

        Returns:
            Path to packaged artifact (tarball or directory)
        """
        print(f"   Packaging workflow: {workflow_id}")

        # Create temporary directory for packaging
        temp_dir = Path(tempfile.mkdtemp(prefix=f'hololoom_{workflow_id}_'))

        try:
            # Step 1: Copy workflow code
            self._copy_workflow_code(workflow_id, temp_dir)

            # Step 2: Generate Dockerfile
            self._generate_dockerfile(workflow_id, temp_dir)

            # Step 3: Generate requirements.txt
            self._generate_requirements(workflow_id, temp_dir, custom_config)

            # Step 4: Generate runtime config
            self._generate_runtime_config(workflow_id, temp_dir)

            # Step 5: Generate environment template
            self._generate_env_template(workflow_id, temp_dir)

            # Step 6: Create health check endpoint
            self._create_health_endpoint(temp_dir)

            # Step 7: Create tarball (optional)
            if custom_config.get('create_tarball', False):
                return self._create_tarball(workflow_id, temp_dir)

            return temp_dir

        except Exception as e:
            # Cleanup on failure
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise Exception(f"Packaging failed: {e}")

    def _copy_workflow_code(self, workflow_id: str, dest_dir: Path) -> None:
        """Copy workflow code to package directory."""
        # Find workflow template
        workflow_file = self._find_workflow_file(workflow_id)

        if workflow_file:
            # Copy workflow Python file
            shutil.copy(workflow_file, dest_dir / 'workflow.py')

            # Copy any dependencies
            workflow_dir = workflow_file.parent
            if (workflow_dir / '__init__.py').exists():
                shutil.copy(workflow_dir / '__init__.py', dest_dir / '__init__.py')

        else:
            # Create minimal workflow file
            (dest_dir / 'workflow.py').write_text(
                f'# Workflow: {workflow_id}\n# Generated: {datetime.now()}\n'
            )

    def _find_workflow_file(self, workflow_id: str) -> Optional[Path]:
        """Find workflow Python file by ID."""
        # Search in workflows/templates directory
        for path in self.workflows_dir.rglob('*.py'):
            if workflow_id in path.stem or workflow_id.replace('-', '_') in path.stem:
                return path

        return None

    def _generate_dockerfile(self, workflow_id: str, dest_dir: Path) -> None:
        """Generate Dockerfile for workflow."""
        dockerfile_content = f"""# Dockerfile for HoloLoom Workflow: {workflow_id}
# Generated: {datetime.now()}

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    postgresql-client \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

        (dest_dir / 'Dockerfile').write_text(dockerfile_content)

    def _generate_requirements(
        self,
        workflow_id: str,
        dest_dir: Path,
        custom_config: Dict[str, Any]
    ) -> None:
        """Generate requirements.txt for workflow."""
        # Base requirements (always needed)
        base_requirements = [
            'fastapi>=0.104.0',
            'uvicorn>=0.24.0',
            'pydantic>=2.5.0',
            'python-multipart>=0.0.6',
        ]

        # Database requirements
        db_requirements = [
            'psycopg2-binary>=2.9.9',
            'sqlalchemy>=2.0.23',
        ]

        # HoloLoom requirements (if using full system)
        hololoom_requirements = [
            'torch>=2.1.0',
            'numpy>=1.24.0',
            'sentence-transformers>=2.2.2',
            'networkx>=3.2.1',
        ]

        # Merge requirements
        all_requirements = base_requirements + db_requirements

        if custom_config.get('use_hololoom_full', False):
            all_requirements.extend(hololoom_requirements)

        # Write requirements.txt
        (dest_dir / 'requirements.txt').write_text('\n'.join(all_requirements) + '\n')

    def _generate_runtime_config(self, workflow_id: str, dest_dir: Path) -> None:
        """Generate runtime configuration."""
        config = {
            'workflow_id': workflow_id,
            'version': '1.0.0',
            'generated_at': datetime.now().isoformat(),
            'python_version': '3.11',
            'port': 8000,
            'settings': {
                'log_level': 'INFO',
                'enable_cors': True,
                'max_workers': 4,
            }
        }

        (dest_dir / 'config.json').write_text(json.dumps(config, indent=2))

    def _generate_env_template(self, workflow_id: str, dest_dir: Path) -> None:
        """Generate .env template for environment variables."""
        env_template = f"""# Environment variables for {workflow_id}
# Copy to .env and fill in your values

# Application
WORKFLOW_ID={workflow_id}
HOLOLOOM_ENV=production
LOG_LEVEL=INFO

# Database (auto-populated by platform)
DATABASE_URL=postgresql://user:pass@localhost:5432/hololoom

# API Keys (fill in your values)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Integrations (workflow-specific)
# GMAIL_API_KEY=your_gmail_key
# SLACK_WEBHOOK_URL=your_slack_webhook
# GITHUB_TOKEN=your_github_token
"""

        (dest_dir / '.env.template').write_text(env_template)

    def _create_health_endpoint(self, dest_dir: Path) -> None:
        """Create main.py with health endpoint."""
        main_content = '''#!/usr/bin/env python3
"""
HoloLoom Workflow Application
Auto-generated deployment application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime

app = FastAPI(
    title="HoloLoom Workflow",
    description="Auto-deployed HoloLoom workflow",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "status": "healthy",
        "workflow": os.getenv("WORKFLOW_ID", "unknown"),
        "platform": os.getenv("HOLOLOOM_PLATFORM", "unknown"),
        "deployed_at": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/config")
async def get_config():
    """Get configuration (non-sensitive)."""
    return {
        "workflow_id": os.getenv("WORKFLOW_ID"),
        "environment": os.getenv("HOLOLOOM_ENV"),
        "log_level": os.getenv("LOG_LEVEL", "INFO")
    }

# Import workflow-specific endpoints
try:
    from workflow import workflow_router
    app.include_router(workflow_router, prefix="/workflow")
except ImportError:
    pass  # Workflow doesn't define custom routes

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
'''

        (dest_dir / 'main.py').write_text(main_content)

    def _create_tarball(self, workflow_id: str, source_dir: Path) -> Path:
        """Create tarball of packaged workflow."""
        tarball_path = source_dir.parent / f'{workflow_id}.tar.gz'

        with tarfile.open(tarball_path, 'w:gz') as tar:
            tar.add(source_dir, arcname=workflow_id)

        return tarball_path

    def list_available_workflows(self) -> List[Dict[str, Any]]:
        """
        List all available workflow templates.

        Returns:
            List of workflow metadata
        """
        workflows = []

        # Scan workflows directory
        for py_file in self.workflows_dir.rglob('*.py'):
            if py_file.stem.startswith('__') or py_file.stem.startswith('test_'):
                continue

            # Extract metadata from file
            try:
                content = py_file.read_text()

                # Simple metadata extraction (look for docstring)
                metadata = {
                    'id': py_file.stem.replace('_', '-'),
                    'name': py_file.stem.replace('_', ' ').title(),
                    'category': py_file.parent.name.title(),
                    'description': self._extract_description(content),
                    'difficulty': 'Beginner',  # Default
                    'estimated_time_saved': self._extract_time_saved(content),
                    'path': str(py_file.relative_to(self.repo_root))
                }

                workflows.append(metadata)

            except Exception:
                continue

        return workflows

    def _extract_description(self, content: str) -> str:
        """Extract description from docstring."""
        lines = content.split('\n')
        for line in lines:
            if 'Automatically' in line or 'Save' in line or 'Impact' in line:
                return line.strip('# " \'')

        return 'HoloLoom workflow template'

    def _extract_time_saved(self, content: str) -> str:
        """Extract time saved from docstring."""
        if 'Save 2 hours/day' in content or '2 hours/day' in content:
            return '2 hours/day'
        elif 'hours' in content.lower():
            return 'Multiple hours/week'
        return 'N/A'
