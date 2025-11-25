#!/usr/bin/env python3
"""
COZ Daily Brief Automation

Automatically generates and saves daily intelligence briefs to file.

Features:
- File-based output (no email dependencies)
- Configurable schedule using Python schedule library
- Error recovery and logging
- Markdown and JSON export
- Optional HoloLoom refinement

Usage:
    # Run once (for testing)
    python daily_brief_automation.py --once

    # Run continuously with schedule (production)
    python daily_brief_automation.py --schedule "09:00"

    # Custom hourly rate and output directory
    python daily_brief_automation.py --schedule "09:00" --hourly-rate 30.0 --output-dir ./briefs

Created: 2025-11-22
Part of: Option 2 - Automation (Week 1)
"""

import argparse
import logging
import schedule
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from elle.coz.sync_manager import SyncManager
from elle.coz.intelligence import IntelligenceEngine


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DailyBriefAutomation:
    """Automated daily brief generation and file export"""

    def __init__(
        self,
        hourly_rate: float = 25.0,
        output_dir: str = "./daily_briefs",
        use_refinement: bool = False,  # Disabled per A/B testing (clarity 1.00 > actionability)
        refinement_provider: str = "anthropic"
    ):
        """
        Initialize automation

        Args:
            hourly_rate: Hourly labor rate for profit calculations
            output_dir: Directory to save brief files
            use_refinement: Use HoloLoom prompt refinement
            refinement_provider: LLM provider ("anthropic", "google", "openai")
        """
        self.hourly_rate = hourly_rate
        self.output_dir = Path(output_dir)
        self.use_refinement = use_refinement
        self.refinement_provider = refinement_provider

        # Create output directory if doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Daily Brief Automation initialized")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        logger.info(f"Hourly rate: ${hourly_rate}")
        logger.info(f"Refinement: {'enabled' if use_refinement else 'disabled'}")

    def generate_and_save(self) -> bool:
        """
        Generate daily brief and save to file

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("=" * 70)
            logger.info("Starting daily brief generation...")
            logger.info("=" * 70)

            # Initialize sync manager and parsers
            logger.info("Loading COZ data parsers...")
            sync = SyncManager()
            sync.parse_all()
            logger.info("Loaded COZ parsers successfully")

            # Initialize intelligence engine
            logger.info("Initializing Intelligence Engine...")
            intelligence = IntelligenceEngine(sync, logger=logger)

            # Generate daily brief
            logger.info("Generating daily brief...")
            brief = intelligence.generate_daily_brief(
                hourly_rate=self.hourly_rate,
                use_refinement=self.use_refinement,
                refinement_provider=self.refinement_provider
            )

            # Generate filenames with timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            date_str = brief['date']

            markdown_filename = f"daily_brief_{date_str}.md"
            json_filename = f"daily_brief_{date_str}.json"

            markdown_path = self.output_dir / markdown_filename
            json_path = self.output_dir / json_filename

            # Save Markdown summary
            logger.info(f"Saving Markdown brief: {markdown_path}")
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(brief['summary'])

            # Save full JSON data
            logger.info(f"Saving JSON data: {json_path}")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(brief, f, indent=2, default=str)

            # Log summary
            logger.info("=" * 70)
            logger.info("Daily Brief Generation Complete!")
            logger.info("=" * 70)
            logger.info(f"Date: {brief['date']}")
            logger.info(f"Refinement used: {brief['refinement_used']}")
            logger.info(f"Action items: {len(brief['action_items'])}")
            logger.info("")
            logger.info("Output files:")
            logger.info(f"  Markdown: {markdown_path.absolute()}")
            logger.info(f"  JSON:     {json_path.absolute()}")
            logger.info("")

            # Log key metrics
            perf = brief['performance_indicators']
            logger.info("Performance Indicators:")
            logger.info(f"  Profit Margin: {perf['profit_margin']:.1f}%")
            logger.info(f"  Hourly Profit: ${perf['hourly_profit']:.2f}/hour")
            logger.info(f"  Task Efficiency: {perf.get('task_efficiency', 0.0):.1%}")
            logger.info("")

            # Log top action items
            logger.info("Top 3 Action Items:")
            for i, action in enumerate(brief['action_items'][:3], 1):
                logger.info(f"  {i}. {action}")

            logger.info("=" * 70)

            return True

        except Exception as e:
            logger.error(f"Daily brief generation failed: {e}", exc_info=True)
            return False

    def schedule_daily(self, time_str: str = "09:00"):
        """
        Schedule daily brief generation

        Args:
            time_str: Time to run daily (HH:MM format, e.g., "09:00")
        """
        logger.info(f"Scheduling daily brief generation at {time_str}")

        # Schedule job
        schedule.every().day.at(time_str).do(self.generate_and_save)

        logger.info("Scheduler started. Press Ctrl+C to stop.")
        logger.info("")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("\nScheduler stopped by user")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="COZ Daily Brief Automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run once (testing)
  python daily_brief_automation.py --once

  # Schedule for 9 AM daily (production)
  python daily_brief_automation.py --schedule "09:00"

  # Custom hourly rate and output directory
  python daily_brief_automation.py --schedule "09:00" --hourly-rate 30.0 --output-dir ./briefs

  # Disable refinement (faster, raw metrics)
  python daily_brief_automation.py --once --no-refinement
        """
    )

    # Mode: once or scheduled
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--once',
        action='store_true',
        help='Run once and exit (for testing)'
    )
    mode_group.add_argument(
        '--schedule',
        type=str,
        metavar='TIME',
        help='Schedule daily at TIME (HH:MM format, e.g., "09:00")'
    )

    # Configuration options
    parser.add_argument(
        '--hourly-rate',
        type=float,
        default=25.0,
        help='Hourly labor rate for profit calculations (default: 25.0)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./daily_briefs',
        help='Output directory for brief files (default: ./daily_briefs)'
    )
    parser.add_argument(
        '--refinement',
        action='store_true',
        help='Enable HoloLoom refinement (slower, higher quality). Disabled by default per A/B testing.'
    )
    parser.add_argument(
        '--refinement-provider',
        type=str,
        default='anthropic',
        choices=['anthropic', 'google', 'openai'],
        help='LLM provider for refinement (default: anthropic)'
    )

    args = parser.parse_args()

    # Create automation instance
    automation = DailyBriefAutomation(
        hourly_rate=args.hourly_rate,
        output_dir=args.output_dir,
        use_refinement=args.refinement,  # Disabled by default per A/B testing
        refinement_provider=args.refinement_provider
    )

    # Run based on mode
    if args.once:
        logger.info("Running once...")
        success = automation.generate_and_save()
        exit(0 if success else 1)
    else:
        automation.schedule_daily(time_str=args.schedule)


if __name__ == "__main__":
    main()
