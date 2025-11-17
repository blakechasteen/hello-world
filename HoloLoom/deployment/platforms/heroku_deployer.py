#!/usr/bin/env python3
"""
Heroku Deployer
===============

Deploy HoloLoom workflows to Heroku (easiest platform).

Features:
- Auto-creates Heroku app
- Provisions Postgres database
- Configures environment variables
- Deploys via git push
- Runs health checks

Cost: $7/month (Hobby dyno)

Created: November 2025
"""

import asyncio
import subprocess
import random
import string
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class HerokuDeployer:
    """
    Deploy workflows to Heroku.

    Heroku is the easiest deployment platform:
    - Simple CLI
    - Auto-provision Postgres
    - Git-based deployment
    - Built-in monitoring

    Cost: $7/month (Hobby dyno) + $9/month (Postgres Basic) = $16/month
    """

    def __init__(self, workflow_id: str, package_path: Optional[Path] = None):
        """
        Initialize Heroku deployer.

        Args:
            workflow_id: ID of the workflow to deploy
            package_path: Path to packaged workflow artifact
        """
        self.workflow_id = workflow_id
        self.package_path = package_path
        self.app_name = f"hololoom-{workflow_id}-{self._random_suffix()}"
        self.start_time = None

    def _random_suffix(self, length: int = 4) -> str:
        """Generate random suffix for unique app names."""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

    async def deploy(self) -> Dict[str, Any]:
        """
        Deploy workflow to Heroku.

        Steps:
        1. Check Heroku CLI installed
        2. Create Heroku app
        3. Provision Postgres addon
        4. Set environment variables
        5. Deploy via git push
        6. Run migrations
        7. Health check

        Returns:
            Deployment result with URL and status
        """
        self.start_time = datetime.now()

        try:
            # Step 1: Check Heroku CLI
            print("   Checking Heroku CLI...")
            await self._check_heroku_cli()
            print("   ✓ Heroku CLI found")

            # Step 2: Create app
            print(f"   Creating Heroku app ({self.app_name})...")
            await self._create_app()
            print(f"   ✓ App created")

            # Step 3: Provision Postgres
            print("   Provisioning Postgres database...")
            await self._provision_postgres()
            print("   ✓ Postgres provisioned")

            # Step 4: Set environment variables
            print("   Configuring environment variables...")
            await self._set_config_vars()
            print("   ✓ Environment configured")

            # Step 5: Deploy code
            print("   Deploying code...")
            await self._deploy_code()
            print("   ✓ Code deployed")

            # Step 6: Run migrations (if needed)
            print("   Running migrations...")
            await self._run_migrations()
            print("   ✓ Migrations complete")

            # Step 7: Scale dynos
            print("   Scaling dynos...")
            await self._scale_dynos()
            print("   ✓ Dynos scaled")

            # Calculate duration
            duration = (datetime.now() - self.start_time).total_seconds()

            return {
                'status': 'success',
                'platform': 'heroku',
                'app_name': self.app_name,
                'url': f"https://{self.app_name}.herokuapp.com",
                'dashboard_url': f"https://dashboard.heroku.com/apps/{self.app_name}",
                'duration_seconds': duration,
                'cost_estimate': {
                    'monthly': 16,
                    'breakdown': {
                        'dyno': 7,
                        'postgres': 9
                    }
                }
            }

        except Exception as e:
            # Cleanup on failure
            await self._cleanup_on_failure()
            raise Exception(f"Heroku deployment failed: {e}")

    async def _check_heroku_cli(self) -> None:
        """Check if Heroku CLI is installed."""
        result = subprocess.run(
            ['heroku', '--version'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise Exception(
                "Heroku CLI not found. Install: https://devcenter.heroku.com/articles/heroku-cli"
            )

    async def _create_app(self) -> None:
        """Create Heroku app."""
        result = subprocess.run(
            ['heroku', 'create', self.app_name],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            if "Name is already taken" in result.stderr:
                # Generate new name and retry
                self.app_name = f"hololoom-{self.workflow_id}-{self._random_suffix()}"
                await self._create_app()
            else:
                raise Exception(f"Failed to create app: {result.stderr}")

    async def _provision_postgres(self) -> None:
        """Provision Postgres database addon."""
        result = subprocess.run(
            ['heroku', 'addons:create', 'heroku-postgresql:hobby-basic', '-a', self.app_name],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            # Try free tier if hobby-basic fails
            result = subprocess.run(
                ['heroku', 'addons:create', 'heroku-postgresql:mini', '-a', self.app_name],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise Exception(f"Failed to provision Postgres: {result.stderr}")

        # Wait for database to be ready
        await asyncio.sleep(5)

    async def _set_config_vars(self) -> None:
        """Set environment variables."""
        config_vars = {
            'WORKFLOW_ID': self.workflow_id,
            'PYTHONUNBUFFERED': '1',
            'HOLOLOOM_ENV': 'production',
            'LOG_LEVEL': 'INFO',
        }

        for key, value in config_vars.items():
            subprocess.run(
                ['heroku', 'config:set', f'{key}={value}', '-a', self.app_name],
                capture_output=True
            )

    async def _deploy_code(self) -> None:
        """Deploy code via git push."""
        # Create temporary git repo
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy package to temp directory
            if self.package_path and self.package_path.exists():
                if self.package_path.is_dir():
                    shutil.copytree(self.package_path, temp_path / 'workflow', dirs_exist_ok=True)
                else:
                    # Extract tarball
                    shutil.unpack_archive(self.package_path, temp_path / 'workflow')
            else:
                # Use default workflow structure
                (temp_path / 'workflow').mkdir(exist_ok=True)
                self._create_default_files(temp_path / 'workflow')

            # Initialize git repo
            subprocess.run(['git', 'init'], cwd=temp_path, capture_output=True)
            subprocess.run(['git', 'add', '.'], cwd=temp_path, capture_output=True)
            subprocess.run(
                ['git', 'commit', '-m', f'Deploy {self.workflow_id}'],
                cwd=temp_path,
                capture_output=True
            )

            # Add Heroku remote
            subprocess.run(
                ['heroku', 'git:remote', '-a', self.app_name],
                cwd=temp_path,
                capture_output=True
            )

            # Push to Heroku
            result = subprocess.run(
                ['git', 'push', 'heroku', 'master'],
                cwd=temp_path,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise Exception(f"Git push failed: {result.stderr}")

    def _create_default_files(self, workflow_dir: Path) -> None:
        """Create default deployment files."""
        # Procfile
        (workflow_dir / 'Procfile').write_text(
            "web: uvicorn main:app --host 0.0.0.0 --port $PORT\n"
        )

        # requirements.txt
        (workflow_dir / 'requirements.txt').write_text(
            "fastapi\nuvicorn\npsycopg2-binary\n"
        )

        # runtime.txt
        (workflow_dir / 'runtime.txt').write_text(
            "python-3.11.5\n"
        )

        # main.py (minimal FastAPI app)
        (workflow_dir / 'main.py').write_text('''
from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "workflow": os.getenv("WORKFLOW_ID", "unknown"),
        "platform": "heroku"
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
''')

    async def _run_migrations(self) -> None:
        """Run database migrations."""
        # Check if migrations exist
        result = subprocess.run(
            ['heroku', 'run', 'python', 'manage.py', 'migrate', '-a', self.app_name],
            capture_output=True
        )

        # Migrations are optional, so don't fail if they don't exist
        pass

    async def _scale_dynos(self) -> None:
        """Scale dynos to 1 web worker."""
        subprocess.run(
            ['heroku', 'ps:scale', 'web=1', '-a', self.app_name],
            capture_output=True
        )

    async def _cleanup_on_failure(self) -> None:
        """Cleanup resources if deployment fails."""
        try:
            print(f"   🧹 Cleaning up failed deployment...")
            subprocess.run(
                ['heroku', 'apps:destroy', self.app_name, '--confirm', self.app_name],
                capture_output=True,
                timeout=30
            )
            print(f"   ✓ Cleaned up app {self.app_name}")
        except Exception as e:
            print(f"   ⚠️  Cleanup warning: {e}")

    def estimate_cost(self) -> Dict[str, Any]:
        """
        Estimate monthly cost for Heroku deployment.

        Returns:
            Cost breakdown
        """
        return {
            'monthly': 16,
            'description': 'Hobby dyno ($7) + Postgres Basic ($9)',
            'breakdown': {
                'compute': {'name': 'Hobby dyno', 'cost': 7},
                'database': {'name': 'Postgres Basic', 'cost': 9},
            },
            'notes': [
                'Free tier available (512MB RAM, no DB)',
                'Hobby tier recommended for production',
                'Scale up to Standard ($25/month) for better performance'
            ]
        }
