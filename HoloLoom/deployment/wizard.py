#!/usr/bin/env python3
"""
Interactive Deployment Wizard
============================

Guides users through workflow deployment with a friendly CLI interface.

Steps:
1. Choose workflow
2. Choose platform
3. Configure credentials
4. Customize settings (optional)
5. Review and confirm
6. Deploy
7. Show results

Created: November 2025
"""

import asyncio
from typing import Dict, Any, List, Optional


class DeploymentWizard:
    """Interactive deployment wizard for beginners."""

    @staticmethod
    async def prompt_choice(question: str, options: List[str], default: Optional[int] = None) -> str:
        """
        Prompt user to select from a list of options.

        Args:
            question: Question to ask
            options: List of options
            default: Default option index

        Returns:
            Selected option
        """
        print(f"\n{question}")
        for i, option in enumerate(options, 1):
            marker = " (default)" if default == i else ""
            print(f"  {i}. {option}{marker}")

        while True:
            try:
                choice = input("\nChoice: ").strip()

                if not choice and default:
                    return options[default - 1]

                choice_num = int(choice)
                if 1 <= choice_num <= len(options):
                    return options[choice_num - 1]

                print(f"⚠️  Please enter a number between 1 and {len(options)}")

            except (ValueError, KeyboardInterrupt):
                print("\n⚠️  Invalid input. Please try again.")

    @staticmethod
    async def prompt_text(question: str, default: Optional[str] = None) -> str:
        """
        Prompt user for text input.

        Args:
            question: Question to ask
            default: Default value

        Returns:
            User input
        """
        default_text = f" (default: {default})" if default else ""
        response = input(f"\n{question}{default_text}: ").strip()

        return response if response else (default or "")

    @staticmethod
    async def confirm(question: str, default: bool = True) -> bool:
        """
        Prompt user for yes/no confirmation.

        Args:
            question: Question to ask
            default: Default answer

        Returns:
            True for yes, False for no
        """
        default_text = " (Y/n)" if default else " (y/N)"
        response = input(f"\n{question}{default_text}: ").strip().lower()

        if not response:
            return default

        return response in ['y', 'yes', 'true', '1']

    @staticmethod
    def show_summary(workflow: str, platform: str, cost_estimate: Dict[str, Any]) -> None:
        """
        Show deployment summary before confirmation.

        Args:
            workflow: Workflow ID
            platform: Platform name
            cost_estimate: Cost estimate data
        """
        print("\n" + "="*60)
        print("📋 Deployment Summary")
        print("="*60)
        print(f"\nWorkflow: {workflow}")
        print(f"Platform: {platform.upper()}")
        print(f"\nEstimated Cost: ${cost_estimate['monthly_total']:.2f}/month")

        print("\nBreakdown:")
        for component, cost in cost_estimate['breakdown'].items():
            print(f"  • {component}: {cost}")

        if cost_estimate.get('notes'):
            print("\nNotes:")
            for note in cost_estimate['notes']:
                print(f"  • {note}")

        print("\n" + "="*60)

    @staticmethod
    def show_success(
        dashboard_url: str,
        cost_estimate: Dict[str, Any],
        deployment_time: float
    ) -> None:
        """
        Show success message with next steps.

        Args:
            dashboard_url: URL to deployment dashboard
            cost_estimate: Cost estimate data
            deployment_time: Time taken to deploy (seconds)
        """
        print("\n" + "="*60)
        print("🎉 Deployment Successful!")
        print("="*60)

        print(f"\n📍 Dashboard: {dashboard_url}")
        print(f"💰 Estimated Cost: ${cost_estimate['monthly_total']:.2f}/month")
        print(f"⏱️  Deployment Time: {deployment_time:.1f}s")

        print("\n📝 Next Steps:")
        print("   1. Visit the dashboard to configure credentials")
        print("   2. Run your first workflow execution")
        print("   3. Check analytics to see time saved")

        print("\nHappy automating! 🚀\n")


async def run_deployment_wizard(orchestrator) -> None:
    """
    Run the interactive deployment wizard.

    Args:
        orchestrator: DeploymentOrchestrator instance
    """
    wizard = DeploymentWizard()

    print("\n" + "="*60)
    print("🚀 HoloLoom One-Click Deployment Wizard")
    print("="*60)
    print("\nDeploy your first workflow in <5 minutes!")

    # Step 1: Choose workflow
    print("\n" + "-"*60)
    print("Step 1: Choose Workflow")
    print("-"*60)

    workflows = orchestrator.packager.list_available_workflows()
    workflow_options = [
        f"{w['id']:20s} - {w['description'][:40]}"
        for w in workflows[:10]  # Show top 10
    ]

    if not workflow_options:
        # Provide default options if no templates found
        workflow_options = [
            "inbox-triage        - Save 2 hours/day triaging emails",
            "bug-triage          - Auto-classify and assign bugs",
            "report-generation   - Generate weekly reports automatically"
        ]

    workflow_choice = await wizard.prompt_choice(
        "Which workflow do you want to deploy?",
        workflow_options,
        default=1
    )

    # Extract workflow ID
    workflow_id = workflow_choice.split('-')[0].strip()

    # Step 2: Choose platform
    print("\n" + "-"*60)
    print("Step 2: Choose Platform")
    print("-"*60)

    platform_options = [
        "heroku       - Easiest, $7/month ⭐ RECOMMENDED",
        "railway      - Modern, $5/month free credit",
        "fly          - Global edge, free tier available",
        "local        - Free, runs on your machine"
    ]

    platform_choice = await wizard.prompt_choice(
        "Where do you want to deploy?",
        platform_options,
        default=1
    )

    # Extract platform
    platform = platform_choice.split(' ')[0].strip()

    # Step 3: Get cost estimate
    print("\n" + "-"*60)
    print("Step 3: Cost Estimate")
    print("-"*60)

    cost_estimate = orchestrator.cost_estimator.estimate(platform, workflow_id)

    print(f"\n💰 Estimated monthly cost: ${cost_estimate['monthly_total']:.2f}")
    print("\nBreakdown:")
    for component, cost in cost_estimate['breakdown'].items():
        print(f"  • {component}: {cost}")

    # Step 4: Configure credentials (optional)
    print("\n" + "-"*60)
    print("Step 4: Configure Credentials (Optional)")
    print("-"*60)

    print("\n💡 You can configure credentials after deployment")
    configure_now = await wizard.confirm(
        "Configure credentials now?",
        default=False
    )

    credentials = {}
    if configure_now:
        # Prompt for common credentials
        # (Simplified - real implementation would be workflow-specific)
        print("\nNote: You can also set these as environment variables later")

    # Step 5: Review and confirm
    print("\n" + "-"*60)
    print("Step 5: Review and Confirm")
    print("-"*60)

    wizard.show_summary(workflow_id, platform, cost_estimate)

    confirmed = await wizard.confirm(
        "Deploy now?",
        default=True
    )

    if not confirmed:
        print("\n❌ Deployment cancelled")
        return

    # Step 6: Deploy
    print("\n" + "-"*60)
    print("Step 6: Deploying...")
    print("-"*60)

    try:
        result = await orchestrator.deploy(
            workflow_id=workflow_id,
            platform=platform,
            custom_config=credentials
        )

        # Step 7: Show success
        wizard.show_success(
            dashboard_url=result.get('url', 'N/A'),
            cost_estimate=cost_estimate,
            deployment_time=result.get('deployment_time', 0)
        )

    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        print("\n💡 Troubleshooting:")
        print(f"   - Check that {platform} CLI is installed")
        print("   - Verify your credentials are configured")
        print("   - Try running with --help for more options")
