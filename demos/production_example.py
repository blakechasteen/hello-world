#!/usr/bin/env python3
"""
HoloLoom Phase 3 - Complete Production Integration Example

This example demonstrates:
1. Setting up adaptive learning classifier
2. Integrating with Prometheus metrics
3. Configuring Slack/email alerts
4. Running production workload
5. Background learning loop

Usage:
    python production_example.py

Environment Variables Required:
    SLACK_WEBHOOK_URL - Slack webhook for alerts
    SMTP_USER - Email username
    SMTP_PASSWORD - Email password
    SMTP_TO - Comma-separated recipient emails
"""

import asyncio
import signal
import logging
import sys
import os
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.routing.query_classifier_adaptive import AdaptiveMoonshotClassifier
from HoloLoom.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/production_example.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ProductionHoloLoom:
    """
    Production-ready HoloLoom Phase 3 deployment.

    Integrates all Phase 3 components:
    - Adaptive learning classifier
    - Background learning loop
    - Prometheus metrics export
    - Slack/email alerting
    - Production monitoring
    """

    def __init__(self, data_dir: Path = Path("./data")):
        self.data_dir = data_dir
        self.classifier: Optional[AdaptiveMoonshotClassifier] = None
        self.running = False
        self.query_count = 0

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "logs").mkdir(exist_ok=True)
        (self.data_dir / "patterns").mkdir(exist_ok=True)
        (self.data_dir / "validation").mkdir(exist_ok=True)
        (self.data_dir / "reports").mkdir(exist_ok=True)

        logger.info(f"Initialized with data_dir: {self.data_dir}")

    async def initialize(self):
        """Initialize production classifier."""
        logger.info("="* 60)
        logger.info("Initializing HoloLoom Phase 3 Production System")
        logger.info("=" * 60)

        # Create classifier with adaptive learning
        logger.info("Creating adaptive classifier...")
        self.classifier = AdaptiveMoonshotClassifier(
            enable_semantic_tier=True,
            enable_adaptive_learning=True,
            data_dir=self.data_dir,
            background_learning=True,
            learning_update_interval=3600.0  # 1 hour
        )

        # Start background learning
        logger.info("Starting background learning loop...")
        await self.classifier.start_background_learning()

        logger.info("✅ Initialization complete")
        logger.info("")

    async def classify_query(self, query: str) -> dict:
        """Classify a single query with full tracking."""
        start_time = asyncio.get_event_loop().time()

        try:
            result = self.classifier.classify(query)
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            self.query_count += 1

            return {
                "query": query,
                "complexity": result.complexity.value,
                "confidence": result.confidence,
                "tier_used": result.tier_used,
                "latency_ms": latency_ms,
                "timestamp": datetime.now().isoformat(),
                "query_number": self.query_count
            }

        except Exception as e:
            logger.error(f"Error classifying query '{query}': {e}")
            return None

    async def export_prometheus_metrics(self):
        """Export metrics to Prometheus textfile collector."""
        try:
            metrics = self.classifier.reporter.export_prometheus_metrics()

            # Write to metrics file
            metrics_file = Path("/tmp/hololoom_phase3_metrics.prom")
            with open(metrics_file, 'w') as f:
                f.write(metrics)

            logger.debug(f"Metrics exported to {metrics_file}")

        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")

    async def check_for_regressions(self):
        """Check for regressions and send alerts if detected."""
        try:
            # Get recent validation results
            if not self.classifier.validator.validation_history:
                return  # No validations yet

            latest = self.classifier.validator.validation_history[-1]

            if latest.regression_detected:
                logger.warning("⚠️  REGRESSION DETECTED")

                # Get alert data
                alert = self.classifier.validator.alerts[-1]

                alert_data = {
                    "title": "Classifier Regression Detected",
                    "current_accuracy": alert.current_accuracy,
                    "baseline_accuracy": alert.baseline_accuracy,
                    "drop_percentage": alert.drop_percentage,
                    "affected_complexity": alert.affected_complexity,
                    "severity": alert.severity,
                    "timestamp": datetime.fromtimestamp(alert.timestamp).isoformat()
                }

                # Send Slack alert
                await self.send_slack_alert(alert_data)

                # Send email alert
                await self.send_email_alert(alert_data)

        except Exception as e:
            logger.error(f"Error checking for regressions: {e}")

    async def send_slack_alert(self, alert_data: dict):
        """Send alert to Slack."""
        slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        if not slack_webhook:
            logger.warning("SLACK_WEBHOOK_URL not set, skipping Slack alert")
            return

        try:
            script_path = Path(__file__).parent / "scripts" / "send_slack_alert.py"
            result = subprocess.run(
                [sys.executable, str(script_path), json.dumps(alert_data)],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                logger.info("✅ Slack alert sent")
            else:
                logger.error(f"❌ Slack alert failed: {result.stderr}")

        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")

    async def send_email_alert(self, alert_data: dict):
        """Send alert via email."""
        smtp_user = os.getenv('SMTP_USER')
        if not smtp_user:
            logger.warning("SMTP_USER not set, skipping email alert")
            return

        try:
            script_path = Path(__file__).parent / "scripts" / "send_email_alert.py"
            result = subprocess.run(
                [sys.executable, str(script_path), json.dumps(alert_data)],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                logger.info("✅ Email alert sent")
            else:
                logger.error(f"❌ Email alert failed: {result.stderr}")

        except Exception as e:
            logger.error(f"Error sending email alert: {e}")

    async def run_production_workload(self):
        """Simulate production workload."""
        logger.info("Starting production workload simulation...")
        logger.info("")

        # Example queries representing different complexities
        example_queries = [
            ("hi", "trivial"),
            ("thanks", "trivial"),
            ("hello there", "trivial"),
            ("what is Python?", "simple"),
            ("define machine learning", "simple"),
            ("explain transformers", "simple"),
            ("how do neural networks work?", "complex"),
            ("describe attention mechanism in detail", "complex"),
            ("compare CNNs and RNNs", "complex"),
            ("analyze tradeoffs of Thompson Sampling vs epsilon-greedy", "research"),
            ("evaluate multi-agent reinforcement learning approaches", "research"),
        ]

        query_index = 0
        check_interval = 10  # Check for regressions every 10 queries

        while self.running:
            # Get next query (cycle through examples)
            query, expected_complexity = example_queries[query_index % len(example_queries)]
            query_index += 1

            # Classify
            result = await self.classify_query(query)

            if result:
                logger.info(f"[{result['query_number']}] Query: \"{query}\"")
                logger.info(f"     Complexity: {result['complexity']} (expected: {expected_complexity})")
                logger.info(f"     Confidence: {result['confidence']:.1%}")
                logger.info(f"     Latency: {result['latency_ms']:.1f}ms")
                logger.info(f"     Tier: {result['tier_used']}")
                logger.info("")

            # Export metrics periodically
            if self.query_count % 5 == 0:
                await self.export_prometheus_metrics()

            # Check for regressions periodically
            if self.query_count % check_interval == 0:
                await self.check_for_regressions()

            # Show statistics periodically
            if self.query_count % 20 == 0:
                await self.show_statistics()

            # Delay between queries
            await asyncio.sleep(2)  # 0.5 QPS

    async def show_statistics(self):
        """Show current learning statistics."""
        try:
            stats = self.classifier.get_learning_statistics()

            logger.info("=" * 60)
            logger.info("📊 Current Statistics")
            logger.info("=" * 60)
            logger.info(f"Total Queries: {stats['total_queries_logged']}")
            logger.info(f"Patterns Discovered: {stats['patterns_discovered']}")
            logger.info(f"Patterns Deployed: {stats['patterns_deployed']}")
            logger.info(f"Validation Accuracy: {stats['validation_accuracy']:.1%}")
            logger.info(f"Regression Alerts: {stats['regression_alerts']}")
            logger.info("=" * 60)
            logger.info("")

        except Exception as e:
            logger.error(f"Error showing statistics: {e}")

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("Shutting down production system...")
        logger.info("=" * 60)

        self.running = False

        if self.classifier:
            # Stop background learning
            logger.info("Stopping background learning...")
            await self.classifier.stop_background_learning()

            # Export final metrics
            logger.info("Exporting final metrics...")
            await self.export_prometheus_metrics()

            # Show final statistics
            logger.info("")
            await self.show_statistics()

        logger.info("✅ Shutdown complete")
        logger.info("")

    async def run(self):
        """Main production loop."""
        self.running = True

        try:
            # Run workload
            await self.run_production_workload()

        except KeyboardInterrupt:
            logger.info("\nReceived interrupt signal")

        except Exception as e:
            logger.error(f"Production error: {e}")

        finally:
            await self.shutdown()

# Global app instance
app: Optional[ProductionHoloLoom] = None

def signal_handler(sig, frame):
    """Handle shutdown signals."""
    global app
    logger.info(f"\nReceived signal {sig}")
    if app:
        app.running = False

async def main():
    """Main entry point."""
    global app

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and initialize app
    app = ProductionHoloLoom(data_dir=Path("./data"))
    await app.initialize()

    # Run production loop
    await app.run()

if __name__ == "__main__":
    asyncio.run(main())
