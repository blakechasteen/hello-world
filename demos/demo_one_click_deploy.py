#!/usr/bin/env python3
"""
One-Click Deployment Demo
========================

Demonstrates the one-click deployment system in action.

Shows:
1. Heroku deployment (easiest)
2. Railway deployment (modern)
3. Local Docker deployment (fallback)
4. Cost comparisons
5. Platform features

Usage:
    python demos/demo_one_click_deploy.py

Created: November 2025
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.deployment.one_click_deploy import DeploymentOrchestrator
from HoloLoom.deployment.cost_estimator import CostEstimator


async def demo_heroku_deployment():
    """Demo 1: Deploy to Heroku (easiest platform)."""
    print("\n" + "="*80)
    print("DEMO 1: Heroku Deployment (Easiest)")
    print("="*80)

    print("\nScenario: Deploy 'inbox-triage' workflow to Heroku")
    print("Expected time: ~3 minutes")
    print("Expected cost: $16/month")

    print("\nWhat happens:")
    print("  1. Check Heroku CLI installed")
    print("  2. Create Heroku app (hololoom-inbox-triage-XXXX)")
    print("  3. Provision Postgres database (Hobby Basic)")
    print("  4. Set environment variables")
    print("  5. Deploy code via git push")
    print("  6. Scale dynos to 1 web worker")
    print("  7. Run health checks")

    print("\nResult:")
    print("  ✓ App URL: https://hololoom-inbox-triage-a8f3.herokuapp.com")
    print("  ✓ Dashboard: https://dashboard.heroku.com/apps/hololoom-inbox-triage-a8f3")
    print("  ✓ Cost: $16/month ($7 dyno + $9 database)")
    print("  ✓ Deployment time: 2 min 47s")

    print("\n💡 Command to run:")
    print("  python HoloLoom/deployment/one_click_deploy.py \\")
    print("    --workflow inbox-triage \\")
    print("    --platform heroku")


async def demo_railway_deployment():
    """Demo 2: Deploy to Railway (modern platform)."""
    print("\n" + "="*80)
    print("DEMO 2: Railway Deployment (Modern)")
    print("="*80)

    print("\nScenario: Deploy 'bug-triage' workflow to Railway")
    print("Expected time: ~2 minutes")
    print("Expected cost: $5/month (free credit covers this)")

    print("\nWhat happens:")
    print("  1. Check Railway CLI installed")
    print("  2. Login to Railway")
    print("  3. Create Railway project")
    print("  4. Add Postgres database")
    print("  5. Deploy via railway up")
    print("  6. Wait for build and deployment")
    print("  7. Run health checks")

    print("\nResult:")
    print("  ✓ App URL: https://hololoom-bug-triage.up.railway.app")
    print("  ✓ Dashboard: https://railway.app/dashboard")
    print("  ✓ Cost: $5/month (covered by $5 free credit)")
    print("  ✓ Deployment time: 1 min 52s")

    print("\n💡 Command to run:")
    print("  python HoloLoom/deployment/one_click_deploy.py \\")
    print("    --workflow bug-triage \\")
    print("    --platform railway")


async def demo_local_docker_deployment():
    """Demo 3: Deploy locally with Docker (fallback)."""
    print("\n" + "="*80)
    print("DEMO 3: Local Docker Deployment (Fallback)")
    print("="*80)

    print("\nScenario: Deploy 'report-generation' workflow locally")
    print("Expected time: ~1 minute")
    print("Expected cost: $0 (free)")

    print("\nWhat happens:")
    print("  1. Check Docker installed and running")
    print("  2. Create docker-compose.yml")
    print("  3. Create Dockerfile")
    print("  4. Build Docker images")
    print("  5. Start containers (web + postgres)")
    print("  6. Wait for services to be healthy")
    print("  7. Run health checks")

    print("\nResult:")
    print("  ✓ App URL: http://localhost:8000")
    print("  ✓ Analytics: http://localhost:8000/analytics")
    print("  ✓ Cost: $0 (runs on your machine)")
    print("  ✓ Deployment time: 58s")

    print("\n💡 Command to run:")
    print("  python HoloLoom/deployment/one_click_deploy.py \\")
    print("    --workflow report-generation \\")
    print("    --platform local")


async def demo_cost_comparison():
    """Demo 4: Cost comparison across all platforms."""
    print("\n" + "="*80)
    print("DEMO 4: Cost Comparison Across All Platforms")
    print("="*80)

    estimator = CostEstimator()

    # Compare costs for inbox-triage workflow
    workflow_id = "inbox-triage"
    comparison = estimator.compare_platforms(workflow_id)

    print(f"\nWorkflow: {workflow_id}")
    print("\nPlatform Comparison (Monthly Costs):\n")

    print(f"{'Platform':<15} {'Monthly':<12} {'Annual':<12} {'Notes'}")
    print("-" * 80)

    for platform, data in comparison['platforms'].items():
        monthly = data['monthly_total']
        annual = data['annual_total']
        notes = data['notes'][0] if data['notes'] else ''

        print(f"{platform:<15} ${monthly:<11.2f} ${annual:<11.2f} {notes[:40]}")

    print("\nRecommendation:")
    print(f"  🥇 Cheapest: {comparison['cheapest'].upper()} "
          f"(${comparison['cost_range']['min']:.2f}/month)")
    print(f"  💡 Best for beginners: HEROKU (easiest setup)")
    print(f"  💡 Best for production: RAILWAY (modern, good value)")


async def demo_interactive_wizard():
    """Demo 5: Interactive deployment wizard."""
    print("\n" + "="*80)
    print("DEMO 5: Interactive Deployment Wizard")
    print("="*80)

    print("\nThe interactive wizard guides you through deployment step-by-step.")
    print("\nExample interaction:")

    print("\n" + "-"*80)
    print("🚀 HoloLoom One-Click Deployment Wizard")
    print("-"*80)

    print("\nStep 1: Choose workflow")
    print("  1. inbox-triage        - Save 2 hours/day triaging emails")
    print("  2. bug-triage          - Auto-classify and assign bugs")
    print("  3. report-generation   - Generate weekly reports")
    print("\nChoice: 1")

    print("\nStep 2: Choose platform")
    print("  1. heroku       - Easiest, $7/month ⭐ RECOMMENDED")
    print("  2. railway      - Modern, $5/month free credit")
    print("  3. fly          - Global edge, free tier")
    print("  4. local        - Free, runs on your machine")
    print("\nChoice: 1")

    print("\nStep 3: Cost estimate")
    print("💰 Estimated monthly cost: $16.00")
    print("  • Compute: $7.00")
    print("  • Database: $9.00")
    print("  • LLM API: $3.00")

    print("\nStep 4: Review and confirm")
    print("="*60)
    print("📋 Deployment Summary")
    print("="*60)
    print("Workflow: inbox-triage")
    print("Platform: HEROKU")
    print("Estimated Cost: $16.00/month")
    print("\nDeploy now? (Y/n): Y")

    print("\n🚀 Deploying...")
    print("  ✓ Heroku CLI found")
    print("  ✓ App created")
    print("  ✓ Postgres provisioned")
    print("  ✓ Environment configured")
    print("  ✓ Code deployed")

    print("\n🎉 Deployment Successful!")
    print("📍 Dashboard: https://hololoom-inbox-triage-a8f3.herokuapp.com")

    print("\n💡 Command to run:")
    print("  python HoloLoom/deployment/one_click_deploy.py --interactive")


async def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("HoloLoom One-Click Deployment System - Demo")
    print("="*80)

    print("\nThis demo shows how to deploy workflows to production in <5 minutes.")
    print("\nWe'll demonstrate:")
    print("  1. Heroku deployment (easiest)")
    print("  2. Railway deployment (modern)")
    print("  3. Local Docker deployment (free)")
    print("  4. Cost comparison across platforms")
    print("  5. Interactive deployment wizard")

    input("\nPress Enter to continue...")

    # Run all demos
    await demo_heroku_deployment()
    input("\nPress Enter for next demo...")

    await demo_railway_deployment()
    input("\nPress Enter for next demo...")

    await demo_local_docker_deployment()
    input("\nPress Enter for next demo...")

    await demo_cost_comparison()
    input("\nPress Enter for next demo...")

    await demo_interactive_wizard()

    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80)

    print("\n📚 Next Steps:")
    print("  1. Install platform CLI (heroku, railway, flyctl, aws, or docker)")
    print("  2. Run: python HoloLoom/deployment/one_click_deploy.py --interactive")
    print("  3. Deploy your first workflow in <5 minutes!")

    print("\n📖 Documentation:")
    print("  • README: HoloLoom/deployment/README.md")
    print("  • Platform comparison: HoloLoom/deployment/PLATFORM_COMPARISON.md")
    print("  • Troubleshooting: HoloLoom/deployment/TROUBLESHOOTING.md")

    print("\nHappy automating! 🚀\n")


if __name__ == '__main__':
    asyncio.run(main())
