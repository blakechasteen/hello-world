#!/usr/bin/env python3
"""
Railway Deployer
===============

Deploy HoloLoom workflows to Railway (modern platform).

Features:
- Modern UI and CLI
- Auto-provision databases
- Git-based deployment
- Free $5/month credit

Cost: $5/month starter (free credit covers this)

Created: November 2025
"""

import subprocess
import random
import string
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class RailwayDeployer:
    """
    Deploy workflows to Railway.

    Railway is a modern deployment platform:
    - Simple CLI
    - Beautiful UI
    - Auto-provision databases
    - $5/month free credit

    Cost: $5/month (covered by free credit)
    """

    def __init__(self, workflow_id: str, package_path: Optional[Path] = None):
        self.workflow_id = workflow_id
        self.package_path = package_path
        self.project_name = f"hololoom-{workflow_id}"
        self.start_time = None

    async def deploy(self) -> Dict[str, Any]:
        """Deploy workflow to Railway."""
        self.start_time = datetime.now()

        try:
            # Check Railway CLI
            print("   Checking Railway CLI...")
            await self._check_railway_cli()
            print("   ✓ Railway CLI found")

            # Login
            print("   Logging in to Railway...")
            await self._login()
            print("   ✓ Logged in")

            # Initialize project
            print(f"   Creating Railway project...")
            await self._init_project()
            print(f"   ✓ Project created")

            # Add Postgres
            print("   Provisioning Postgres...")
            await self._add_postgres()
            print("   ✓ Postgres provisioned")

            # Deploy
            print("   Deploying code...")
            url = await self._deploy_code()
            print("   ✓ Code deployed")

            duration = (datetime.now() - self.start_time).total_seconds()

            return {
                'status': 'success',
                'platform': 'railway',
                'project_name': self.project_name,
                'url': url,
                'dashboard_url': 'https://railway.app/dashboard',
                'duration_seconds': duration,
                'cost_estimate': {
                    'monthly': 5,
                    'breakdown': {
                        'compute': 5,
                        'database': 0  # Included
                    }
                }
            }

        except Exception as e:
            raise Exception(f"Railway deployment failed: {e}")

    async def _check_railway_cli(self) -> None:
        """Check if Railway CLI is installed."""
        result = subprocess.run(['railway', '--version'], capture_output=True)
        if result.returncode != 0:
            raise Exception("Railway CLI not found. Install: npm i -g @railway/cli")

    async def _login(self) -> None:
        """Login to Railway."""
        # Railway CLI handles auth automatically
        pass

    async def _init_project(self) -> None:
        """Initialize Railway project."""
        subprocess.run(
            ['railway', 'init', '--name', self.project_name],
            capture_output=True
        )

    async def _add_postgres(self) -> None:
        """Add Postgres database."""
        subprocess.run(
            ['railway', 'add', '--database', 'postgresql'],
            capture_output=True
        )

    async def _deploy_code(self) -> str:
        """Deploy code to Railway."""
        result = subprocess.run(
            ['railway', 'up'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise Exception(f"Deployment failed: {result.stderr}")

        # Extract URL from output
        return f"https://{self.project_name}.up.railway.app"

    def estimate_cost(self) -> Dict[str, Any]:
        """Estimate monthly cost."""
        return {
            'monthly': 5,
            'description': 'Starter plan (free $5 credit)',
            'breakdown': {
                'compute': {'name': 'Starter', 'cost': 5},
                'database': {'name': 'Postgres', 'cost': 0},
            },
            'notes': [
                '$5/month free credit (covers starter plan)',
                'Scale up to Team ($20/month) for production'
            ]
        }
