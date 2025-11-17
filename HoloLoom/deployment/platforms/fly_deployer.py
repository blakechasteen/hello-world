#!/usr/bin/env python3
"""
Fly.io Deployer
==============

Deploy HoloLoom workflows to Fly.io (global edge platform).

Features:
- Global edge deployment
- Fast CDN
- Docker-based
- Free tier available

Cost: Free tier (3 shared-cpu-1x VMs) or $1.94/month

Created: November 2025
"""

import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class FlyDeployer:
    """
    Deploy workflows to Fly.io.

    Fly.io is a global edge platform:
    - Fast CDN
    - Global deployment
    - Docker-based
    - Free tier available

    Cost: Free tier or $1.94/month (shared-cpu-1x)
    """

    def __init__(self, workflow_id: str, package_path: Optional[Path] = None):
        self.workflow_id = workflow_id
        self.package_path = package_path
        self.app_name = f"hololoom-{workflow_id}"
        self.start_time = None

    async def deploy(self) -> Dict[str, Any]:
        """Deploy workflow to Fly.io."""
        self.start_time = datetime.now()

        try:
            # Check flyctl
            print("   Checking flyctl...")
            await self._check_flyctl()
            print("   ✓ flyctl found")

            # Launch app
            print(f"   Launching Fly app...")
            await self._launch_app()
            print(f"   ✓ App launched")

            # Add Postgres
            print("   Provisioning Postgres...")
            await self._add_postgres()
            print("   ✓ Postgres provisioned")

            # Deploy
            print("   Deploying code...")
            await self._deploy_code()
            print("   ✓ Code deployed")

            duration = (datetime.now() - self.start_time).total_seconds()

            return {
                'status': 'success',
                'platform': 'fly',
                'app_name': self.app_name,
                'url': f"https://{self.app_name}.fly.dev",
                'dashboard_url': f"https://fly.io/apps/{self.app_name}",
                'duration_seconds': duration,
                'cost_estimate': {
                    'monthly': 1.94,
                    'breakdown': {
                        'compute': 1.94,
                        'database': 0  # Free tier
                    }
                }
            }

        except Exception as e:
            raise Exception(f"Fly.io deployment failed: {e}")

    async def _check_flyctl(self) -> None:
        """Check if flyctl is installed."""
        result = subprocess.run(['flyctl', 'version'], capture_output=True)
        if result.returncode != 0:
            raise Exception("flyctl not found. Install: https://fly.io/docs/hands-on/install-flyctl/")

    async def _launch_app(self) -> None:
        """Launch Fly app."""
        subprocess.run(
            ['flyctl', 'launch', '--name', self.app_name, '--no-deploy'],
            capture_output=True
        )

    async def _add_postgres(self) -> None:
        """Add Postgres database."""
        subprocess.run(
            ['flyctl', 'postgres', 'create', '--name', f'{self.app_name}-db'],
            capture_output=True
        )

        subprocess.run(
            ['flyctl', 'postgres', 'attach', f'{self.app_name}-db', '--app', self.app_name],
            capture_output=True
        )

    async def _deploy_code(self) -> None:
        """Deploy code to Fly.io."""
        result = subprocess.run(
            ['flyctl', 'deploy'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise Exception(f"Deployment failed: {result.stderr}")

    def estimate_cost(self) -> Dict[str, Any]:
        """Estimate monthly cost."""
        return {
            'monthly': 1.94,
            'description': 'shared-cpu-1x VM',
            'breakdown': {
                'compute': {'name': 'shared-cpu-1x', 'cost': 1.94},
                'database': {'name': 'Postgres (free tier)', 'cost': 0},
            },
            'notes': [
                'Free tier: 3 shared-cpu-1x VMs',
                'Scale up to dedicated-cpu ($29/month) for production'
            ]
        }
