#!/usr/bin/env python3
"""
HoloLoom One-Click Deployment System
====================================

Deploy HoloLoom workflows to production in <5 minutes with zero infrastructure knowledge.

Supported Platforms:
- Heroku (easiest, $7/month)
- Railway (modern, $5/month credit)
- Fly.io (global edge, free tier)
- AWS Lambda (serverless, pay-per-use)
- Local Docker (free, always works)

Usage:
    # Interactive mode (recommended for beginners)
    python one_click_deploy.py --interactive

    # Direct deployment
    python one_click_deploy.py --workflow inbox-triage --platform heroku

    # Show available workflows
    python one_click_deploy.py --list-workflows

    # Show available platforms
    python one_click_deploy.py --list-platforms

Created: November 2025
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from HoloLoom.deployment.platforms.heroku_deployer import HerokuDeployer
    from HoloLoom.deployment.platforms.railway_deployer import RailwayDeployer
    from HoloLoom.deployment.platforms.fly_deployer import FlyDeployer
    from HoloLoom.deployment.platforms.aws_lambda_deployer import AWSLambdaDeployer
    from HoloLoom.deployment.platforms.local_docker_deployer import LocalDockerDeployer
    from HoloLoom.deployment.wizard import run_deployment_wizard
    from HoloLoom.deployment.packager import WorkflowPackager
    from HoloLoom.deployment.health_checker import HealthChecker
    from HoloLoom.deployment.cost_estimator import CostEstimator
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("Make sure you're running from the repository root with PYTHONPATH set")
    sys.exit(1)


class DeploymentOrchestrator:
    """
    Main orchestrator for one-click workflow deployment.

    Handles platform selection, deployment coordination, and post-deployment verification.
    """

    def __init__(self):
        self.packager = WorkflowPackager()
        self.health_checker = HealthChecker()
        self.cost_estimator = CostEstimator()

        # Platform deployer registry
        self.deployers = {
            'heroku': HerokuDeployer,
            'railway': RailwayDeployer,
            'fly': FlyDeployer,
            'aws_lambda': AWSLambdaDeployer,
            'aws': AWSLambdaDeployer,  # Alias
            'local': LocalDockerDeployer,
            'docker': LocalDockerDeployer,  # Alias
        }

    async def deploy(
        self,
        workflow_id: str,
        platform: str,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Deploy a workflow to the specified platform.

        Args:
            workflow_id: ID of the workflow template (e.g., "inbox-triage")
            platform: Deployment platform (heroku/railway/fly/aws/local)
            custom_config: Optional custom configuration

        Returns:
            Deployment result with URL, status, and metrics
        """
        print(f"\n🚀 Deploying '{workflow_id}' to {platform.upper()}...\n")

        # Validate platform
        if platform.lower() not in self.deployers:
            available = ', '.join(self.deployers.keys())
            raise ValueError(f"Unknown platform '{platform}'. Available: {available}")

        # Step 1: Package workflow
        print("📦 Packaging workflow...")
        package_path = self.packager.package(workflow_id, custom_config or {})
        print(f"   ✓ Packaged to {package_path}")

        # Step 2: Estimate cost
        print("\n💰 Estimating costs...")
        cost_estimate = self.cost_estimator.estimate(platform, workflow_id)
        print(f"   Monthly cost: ${cost_estimate['monthly_total']:.2f}")
        print(f"   Breakdown: Base ${cost_estimate['monthly_base']:.2f} + "
              f"DB ${cost_estimate.get('monthly_database', 0):.2f} + "
              f"LLM ~${cost_estimate.get('monthly_llm', 0):.2f}")

        # Step 3: Create deployer and deploy
        print(f"\n🚢 Deploying to {platform.upper()}...")
        deployer_class = self.deployers[platform.lower()]
        deployer = deployer_class(workflow_id, package_path)

        try:
            deployment_result = await deployer.deploy()

            # Step 4: Health check
            if deployment_result.get('url'):
                print("\n🏥 Running health checks...")
                health_status = await self.health_checker.check_deployment(
                    deployment_result['url']
                )

                if health_status.healthy:
                    print("   ✓ All health checks passed")
                else:
                    print("   ⚠️  Some health checks failed:")
                    for check in health_status.checks:
                        if not check['passed']:
                            print(f"      - {check['name']}: {check['message']}")

            # Step 5: Show success
            print("\n" + "="*60)
            print("🎉 Deployment Successful!")
            print("="*60)

            if deployment_result.get('url'):
                print(f"\n📍 Dashboard: {deployment_result['url']}")

            print(f"\n💰 Estimated Cost: ${cost_estimate['monthly_total']:.2f}/month")
            print(f"⏱️  Deployment Time: {deployment_result.get('duration_seconds', 0):.1f}s")

            print("\n📝 Next Steps:")
            print("   1. Visit the dashboard to configure credentials")
            print("   2. Run your first workflow execution")
            print("   3. Check analytics to see time saved")

            print("\nHappy automating! 🚀\n")

            return {
                'status': 'success',
                'platform': platform,
                'workflow_id': workflow_id,
                'url': deployment_result.get('url'),
                'cost_estimate': cost_estimate,
                'deployment_time': deployment_result.get('duration_seconds'),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"\n❌ Deployment failed: {e}")
            print("\n💡 Troubleshooting:")
            print(f"   - Check that {platform} CLI is installed")
            print("   - Verify your credentials are configured")
            print("   - See logs above for detailed error")

            raise

    def list_workflows(self) -> None:
        """List all available workflow templates."""
        print("\n📚 Available Workflow Templates:\n")

        workflows = self.packager.list_available_workflows()

        # Group by category
        by_category = {}
        for wf in workflows:
            category = wf.get('category', 'Other')
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(wf)

        for category, wfs in sorted(by_category.items()):
            print(f"📁 {category}:")
            for wf in wfs:
                print(f"   • {wf['id']:20s} - {wf['name']}")
                print(f"     {wf['description']}")
                print(f"     Impact: {wf.get('estimated_time_saved', 'N/A')} | "
                      f"Difficulty: {wf.get('difficulty', 'N/A')}")
                print()

        print(f"Total: {len(workflows)} workflows\n")

    def list_platforms(self) -> None:
        """List all supported deployment platforms."""
        print("\n🌐 Supported Deployment Platforms:\n")

        platforms = [
            {
                'name': 'Heroku',
                'id': 'heroku',
                'difficulty': '⭐ Easiest',
                'cost': '$7/month (Hobby)',
                'features': ['Auto-provision Postgres', 'Simple git push deploy', 'Built-in monitoring'],
                'best_for': 'Beginners, quick prototypes'
            },
            {
                'name': 'Railway',
                'id': 'railway',
                'difficulty': '⭐⭐ Easy',
                'cost': '$5/month credit (free to start)',
                'features': ['Modern UI', 'Auto-provision databases', 'Git-based deployment'],
                'best_for': 'Modern developers, side projects'
            },
            {
                'name': 'Fly.io',
                'id': 'fly',
                'difficulty': '⭐⭐⭐ Medium',
                'cost': 'Free tier available',
                'features': ['Global edge deployment', 'Fast CDN', 'Docker-based'],
                'best_for': 'Global applications, low latency'
            },
            {
                'name': 'AWS Lambda',
                'id': 'aws_lambda',
                'difficulty': '⭐⭐⭐⭐ Advanced',
                'cost': 'Pay per use (~$0.20/1M requests)',
                'features': ['Serverless', 'Auto-scaling', 'AWS ecosystem integration'],
                'best_for': 'Enterprise, high scale'
            },
            {
                'name': 'Local Docker',
                'id': 'local',
                'difficulty': '⭐ Easiest',
                'cost': 'Free',
                'features': ['Runs on your machine', 'No cloud account needed', 'docker-compose based'],
                'best_for': 'Testing, development, offline use'
            }
        ]

        for p in platforms:
            print(f"{'='*60}")
            print(f"{p['name']} ({p['id']})")
            print(f"{'='*60}")
            print(f"Difficulty: {p['difficulty']}")
            print(f"Cost: {p['cost']}")
            print(f"Best for: {p['best_for']}")
            print(f"Features:")
            for feature in p['features']:
                print(f"  • {feature}")
            print()

        print("💡 Recommendation: Start with Heroku (easiest) or Local Docker (free)\n")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy HoloLoom workflows in <5 minutes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (beginner-friendly)
  python one_click_deploy.py --interactive

  # Deploy to Heroku
  python one_click_deploy.py --workflow inbox-triage --platform heroku

  # Deploy to Railway
  python one_click_deploy.py --workflow bug-triage --platform railway

  # Deploy locally (Docker)
  python one_click_deploy.py --workflow report-generation --platform local

  # List available workflows
  python one_click_deploy.py --list-workflows

  # List supported platforms
  python one_click_deploy.py --list-platforms
        """
    )

    parser.add_argument(
        '--workflow',
        type=str,
        help='Workflow ID to deploy (e.g., inbox-triage)'
    )

    parser.add_argument(
        '--platform',
        type=str,
        choices=['heroku', 'railway', 'fly', 'aws_lambda', 'aws', 'local', 'docker'],
        help='Deployment platform'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run interactive deployment wizard (recommended for beginners)'
    )

    parser.add_argument(
        '--list-workflows',
        action='store_true',
        help='List all available workflow templates'
    )

    parser.add_argument(
        '--list-platforms',
        action='store_true',
        help='List all supported deployment platforms'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file (JSON)'
    )

    args = parser.parse_args()

    orchestrator = DeploymentOrchestrator()

    # Handle list commands
    if args.list_workflows:
        orchestrator.list_workflows()
        return

    if args.list_platforms:
        orchestrator.list_platforms()
        return

    # Interactive mode
    if args.interactive:
        print("\n🚀 HoloLoom One-Click Deployment Wizard\n")
        await run_deployment_wizard(orchestrator)
        return

    # Direct deployment mode
    if not args.workflow or not args.platform:
        parser.print_help()
        print("\n💡 Tip: Use --interactive for a guided experience")
        return

    # Load custom config if provided
    custom_config = None
    if args.config:
        import json
        with open(args.config) as f:
            custom_config = json.load(f)

    # Deploy
    try:
        result = await orchestrator.deploy(
            workflow_id=args.workflow,
            platform=args.platform,
            custom_config=custom_config
        )

        # Exit with success
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
