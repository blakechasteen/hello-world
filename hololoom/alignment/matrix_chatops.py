"""
Matrix.org ChatOps Integration for Alignment Framework

Sends alignment alerts to Matrix rooms for team visibility and response.

Features:
- Webhook-based notifications (simple)
- Bot client with Matrix SDK (advanced)
- Rich formatted messages (HTML + Markdown)
- Alert severity indicators
- Metric summaries
- Interactive responses (bot mode)

Usage:
    # Simple webhook mode
    from hololoom.alignment.matrix_chatops import send_matrix_webhook

    send_matrix_webhook(
        alert,
        webhook_url="https://matrix.example.com/_matrix/client/r0/rooms/..."
    )

    # Bot mode (requires matrix-nio)
    from hololoom.alignment.matrix_chatops import MatrixBot

    bot = MatrixBot(
        homeserver="https://matrix.org",
        user_id="@bot:matrix.org",
        access_token="syt_..."
    )

    await bot.send_alert(alert, room_id="!room:matrix.org")

Environment Variables:
    MATRIX_HOMESERVER: Matrix homeserver URL
    MATRIX_USER_ID: Bot user ID
    MATRIX_ACCESS_TOKEN: Bot access token
    MATRIX_ROOM_ID: Default room ID for alerts
"""

import os
import json
import logging
from typing import Optional, Dict, List
from datetime import datetime
from enum import Enum

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests not installed - webhook mode unavailable")

try:
    from nio import AsyncClient, RoomSendResponse
    MATRIX_SDK_AVAILABLE = True
except ImportError:
    MATRIX_SDK_AVAILABLE = False
    logging.warning("matrix-nio not installed - bot mode unavailable")

from hololoom.alignment.monitoring import Alert, AlertLevel

logger = logging.getLogger(__name__)


class MessageFormat(Enum):
    """Message format options."""
    PLAIN = "plain"
    HTML = "html"
    MARKDOWN = "markdown"


def format_alert_message(
    alert: Alert,
    format_type: MessageFormat = MessageFormat.HTML,
    include_metrics: bool = True
) -> Dict[str, str]:
    """
    Format alert as Matrix message.

    Returns:
        Dict with 'body' (plaintext) and 'formatted_body' (HTML) keys
    """
    # Severity emoji
    emoji_map = {
        AlertLevel.INFO: "ℹ️",
        AlertLevel.WARNING: "⚠️",
        AlertLevel.CRITICAL: "🔴"
    }
    emoji = emoji_map.get(alert.level, "📊")

    # Plain text version
    plain = f"{emoji} {alert.level.value.upper()}: {alert.component}\n"
    plain += f"{alert.message}\n"
    plain += f"Metric: {alert.metric}\n"
    plain += f"Value: {alert.value:.2f}ms (threshold: {alert.threshold:.2f}ms)\n"
    plain += f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

    # HTML version (rich formatting)
    html = f"<h3>{emoji} {alert.level.value.upper()}: {alert.component}</h3>"
    html += f"<p><strong>{alert.message}</strong></p>"
    html += "<ul>"
    html += f"<li><strong>Metric:</strong> {alert.metric}</li>"
    html += f"<li><strong>Value:</strong> {alert.value:.2f}ms</li>"
    html += f"<li><strong>Threshold:</strong> {alert.threshold:.2f}ms</li>"
    html += f"<li><strong>Exceeded by:</strong> {alert.value - alert.threshold:.2f}ms "
    html += f"({(alert.value / alert.threshold - 1) * 100:.1f}%)</li>"
    html += f"<li><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</li>"
    html += "</ul>"

    # Color coding
    color_map = {
        AlertLevel.INFO: "#0088cc",
        AlertLevel.WARNING: "#ff9900",
        AlertLevel.CRITICAL: "#cc0000"
    }
    color = color_map.get(alert.level, "#666666")

    html = f'<div style="border-left: 4px solid {color}; padding-left: 12px;">{html}</div>'

    return {
        "body": plain,
        "formatted_body": html
    }


def send_matrix_webhook(
    alert: Alert,
    webhook_url: str,
    format_type: MessageFormat = MessageFormat.HTML
) -> bool:
    """
    Send alert via Matrix webhook.

    Args:
        alert: Alert to send
        webhook_url: Matrix webhook URL
        format_type: Message format (HTML recommended)

    Returns:
        True if sent successfully, False otherwise

    Note:
        Requires 'requests' library: pip install requests
    """
    if not REQUESTS_AVAILABLE:
        logger.error("requests library not available - install with: pip install requests")
        return False

    try:
        # Format message
        message = format_alert_message(alert, format_type)

        # Build Matrix message payload
        payload = {
            "msgtype": "m.text",
            "body": message["body"],
        }

        # Add HTML formatting if available
        if format_type == MessageFormat.HTML:
            payload["format"] = "org.matrix.custom.html"
            payload["formatted_body"] = message["formatted_body"]

        # Send to webhook
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        response.raise_for_status()
        logger.info(f"Sent alert to Matrix: {alert.component} - {alert.level.value}")
        return True

    except Exception as e:
        logger.error(f"Failed to send Matrix webhook: {e}")
        return False


class MatrixBot:
    """
    Matrix bot for advanced chatops integration.

    Requires matrix-nio: pip install matrix-nio

    Features:
    - Send rich formatted alerts
    - Respond to commands (!status, !metrics)
    - Interactive alert acknowledgment
    - Metric summaries
    """

    def __init__(
        self,
        homeserver: Optional[str] = None,
        user_id: Optional[str] = None,
        access_token: Optional[str] = None,
        default_room_id: Optional[str] = None
    ):
        """
        Initialize Matrix bot.

        Args:
            homeserver: Matrix homeserver URL (e.g., https://matrix.org)
            user_id: Bot user ID (e.g., @bot:matrix.org)
            access_token: Bot access token
            default_room_id: Default room for alerts
        """
        if not MATRIX_SDK_AVAILABLE:
            raise ImportError(
                "matrix-nio not installed. Install with: pip install matrix-nio"
            )

        # Get config from env or params
        self.homeserver = homeserver or os.getenv('MATRIX_HOMESERVER')
        self.user_id = user_id or os.getenv('MATRIX_USER_ID')
        self.access_token = access_token or os.getenv('MATRIX_ACCESS_TOKEN')
        self.default_room_id = default_room_id or os.getenv('MATRIX_ROOM_ID')

        if not all([self.homeserver, self.user_id, self.access_token]):
            raise ValueError(
                "Missing required config. Set MATRIX_HOMESERVER, MATRIX_USER_ID, "
                "and MATRIX_ACCESS_TOKEN environment variables or pass as arguments."
            )

        # Create client
        self.client = AsyncClient(self.homeserver, self.user_id)
        self.client.access_token = self.access_token

        logger.info(f"Matrix bot initialized: {self.user_id} @ {self.homeserver}")

    async def send_alert(
        self,
        alert: Alert,
        room_id: Optional[str] = None
    ) -> bool:
        """
        Send alert to Matrix room.

        Args:
            alert: Alert to send
            room_id: Room ID (uses default if not specified)

        Returns:
            True if sent successfully
        """
        room_id = room_id or self.default_room_id

        if not room_id:
            logger.error("No room_id specified and no default set")
            return False

        try:
            # Format message
            message = format_alert_message(alert, MessageFormat.HTML)

            # Send message
            response = await self.client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": message["body"],
                    "format": "org.matrix.custom.html",
                    "formatted_body": message["formatted_body"]
                }
            )

            if isinstance(response, RoomSendResponse):
                logger.info(f"Sent alert to Matrix room {room_id}: {response.event_id}")
                return True
            else:
                logger.error(f"Failed to send alert: {response}")
                return False

        except Exception as e:
            logger.error(f"Error sending alert to Matrix: {e}")
            return False

    async def send_metrics_summary(
        self,
        stats: Dict,
        room_id: Optional[str] = None
    ) -> bool:
        """
        Send metrics summary to Matrix room.

        Args:
            stats: Stats dict from monitor.get_all_stats()
            room_id: Room ID (uses default if not specified)

        Returns:
            True if sent successfully
        """
        room_id = room_id or self.default_room_id

        if not room_id:
            logger.error("No room_id specified and no default set")
            return False

        try:
            # Build plain text
            plain = "📊 Alignment Framework Metrics\n\n"
            for component, metrics in stats.items():
                if metrics["count"] > 0:
                    plain += f"{component}:\n"
                    plain += f"  P50: {metrics['p50']:.3f}ms\n"
                    plain += f"  P95: {metrics['p95']:.3f}ms\n"
                    plain += f"  P99: {metrics['p99']:.3f}ms\n"
                    plain += f"  Count: {metrics['count']}\n\n"

            # Build HTML
            html = "<h3>📊 Alignment Framework Metrics</h3>"
            html += "<table><tr><th>Component</th><th>P50</th><th>P95</th><th>P99</th><th>Count</th></tr>"

            for component, metrics in stats.items():
                if metrics["count"] > 0:
                    html += f"<tr>"
                    html += f"<td><strong>{component}</strong></td>"
                    html += f"<td>{metrics['p50']:.3f}ms</td>"
                    html += f"<td>{metrics['p95']:.3f}ms</td>"
                    html += f"<td>{metrics['p99']:.3f}ms</td>"
                    html += f"<td>{metrics['count']}</td>"
                    html += f"</tr>"

            html += "</table>"

            # Send message
            response = await self.client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": plain,
                    "format": "org.matrix.custom.html",
                    "formatted_body": html
                }
            )

            return isinstance(response, RoomSendResponse)

        except Exception as e:
            logger.error(f"Error sending metrics summary: {e}")
            return False

    async def close(self):
        """Close Matrix client."""
        await self.client.close()


# Convenience function for quick integration
def setup_matrix_alerting(
    monitor,
    webhook_url: Optional[str] = None,
    bot_config: Optional[Dict] = None,
    check_interval: int = 60
):
    """
    Set up automatic Matrix alerting.

    Args:
        monitor: AlignmentMonitor instance
        webhook_url: Webhook URL for simple mode
        bot_config: Bot config dict for advanced mode
        check_interval: How often to check for alerts (seconds)

    Usage:
        # Webhook mode
        setup_matrix_alerting(
            monitor,
            webhook_url="https://matrix.example.com/..."
        )

        # Bot mode
        setup_matrix_alerting(
            monitor,
            bot_config={
                "homeserver": "https://matrix.org",
                "user_id": "@bot:matrix.org",
                "access_token": "syt_...",
                "room_id": "!room:matrix.org"
            }
        )
    """
    import asyncio
    import threading

    last_alert_count = 0

    async def check_alerts():
        nonlocal last_alert_count

        while True:
            try:
                # Check for new alerts
                current_count = len(monitor.alerts)

                if current_count > last_alert_count:
                    # New alerts
                    new_alerts = monitor.alerts[last_alert_count:]

                    for alert in new_alerts:
                        if alert.level in [AlertLevel.WARNING, AlertLevel.CRITICAL]:
                            # Send to Matrix
                            if webhook_url:
                                send_matrix_webhook(alert, webhook_url)
                            elif bot_config:
                                bot = MatrixBot(**bot_config)
                                await bot.send_alert(alert)
                                await bot.close()

                    last_alert_count = current_count

                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.error(f"Error in alert checker: {e}")
                await asyncio.sleep(check_interval)

    # Start background thread
    def run_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(check_alerts())

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()

    logger.info(f"Matrix alerting started (check interval: {check_interval}s)")


if __name__ == '__main__':
    """
    Test Matrix integration.

    Usage:
        # Set environment variables
        export MATRIX_HOMESERVER=https://matrix.org
        export MATRIX_USER_ID=@bot:matrix.org
        export MATRIX_ACCESS_TOKEN=syt_...
        export MATRIX_ROOM_ID=!room:matrix.org

        python hololoom/alignment/matrix_chatops.py
    """
    import asyncio
    from hololoom.alignment.monitoring import Alert, AlertLevel
    from datetime import datetime

    # Create test alert
    test_alert = Alert(
        level=AlertLevel.CRITICAL,
        component="test_component",
        metric="p99",
        value=15.5,
        threshold=10.0,
        message="P99 latency 15.50ms exceeds critical threshold 10.00ms",
        timestamp=datetime.now()
    )

    print("="*80)
    print("MATRIX.ORG CHATOPS TEST")
    print("="*80)
    print()

    # Test webhook mode
    webhook_url = os.getenv('MATRIX_WEBHOOK_URL')
    if webhook_url and REQUESTS_AVAILABLE:
        print("Testing webhook mode...")
        result = send_matrix_webhook(test_alert, webhook_url)
        print(f"Webhook result: {'✅ Success' if result else '❌ Failed'}")
        print()

    # Test bot mode
    if all([
        os.getenv('MATRIX_HOMESERVER'),
        os.getenv('MATRIX_USER_ID'),
        os.getenv('MATRIX_ACCESS_TOKEN'),
        os.getenv('MATRIX_ROOM_ID')
    ]) and MATRIX_SDK_AVAILABLE:
        print("Testing bot mode...")

        async def test_bot():
            bot = MatrixBot()
            result = await bot.send_alert(test_alert)
            print(f"Bot result: {'✅ Success' if result else '❌ Failed'}")
            await bot.close()

        asyncio.run(test_bot())
    else:
        print("⚠️  Bot mode not configured (set environment variables)")

    print()
    print("="*80)
