#!/usr/bin/env python3
"""
Send Slack alert for HoloLoom Phase 3 regression.

Usage:
    python send_slack_alert.py '{"title": "...", "current_accuracy": 0.85, ...}'

Environment Variables:
    SLACK_WEBHOOK_URL - Required. Slack incoming webhook URL
    SLACK_CHANNEL - Optional. Default: #hololoom-alerts
"""

import os
import sys
import json
import requests
from datetime import datetime
from typing import Dict, Any

SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')
SLACK_CHANNEL = os.getenv('SLACK_CHANNEL', '#hololoom-alerts')

def format_slack_message(alert_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format alert data as Slack message with blocks."""

    severity_emoji = "🚨" if alert_data['severity'] == "CRITICAL" else "⚠️"
    severity_color = "danger" if alert_data['severity'] == "CRITICAL" else "warning"

    message = {
        "channel": SLACK_CHANNEL,
        "username": "HoloLoom Monitor",
        "icon_emoji": ":robot_face:",
        "text": f"{severity_emoji} *{alert_data['title']}*",
        "attachments": [
            {
                "color": severity_color,
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"{severity_emoji} {alert_data['title']}"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Current Accuracy:*\n{alert_data['current_accuracy']:.1%}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Baseline Accuracy:*\n{alert_data['baseline_accuracy']:.1%}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Drop:*\n{alert_data['drop_percentage']:.1%}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Severity:*\n`{alert_data['severity']}`"
                            }
                        ]
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Affected Complexities:*\n{', '.join(alert_data['affected_complexity'])}"
                        }
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Time:* {alert_data['timestamp']} | *Component:* Phase 3 Adaptive Learning"
                            }
                        ]
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "View Dashboard"
                                },
                                "url": "http://localhost:3000/d/hololoom-phase3",
                                "style": "primary"
                            },
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "View Logs"
                                },
                                "url": "http://localhost:9090"
                            }
                        ]
                    }
                ]
            }
        ]
    }

    return message

def send_alert(alert_data: Dict[str, Any]) -> bool:
    """Send formatted alert to Slack."""

    if not SLACK_WEBHOOK_URL:
        print("Error: SLACK_WEBHOOK_URL environment variable not set", file=sys.stderr)
        return False

    # Format message
    message = format_slack_message(alert_data)

    # Send to Slack
    try:
        response = requests.post(
            SLACK_WEBHOOK_URL,
            json=message,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )

        if response.status_code == 200:
            print(f"✅ Slack alert sent successfully to {SLACK_CHANNEL}")
            return True
        else:
            print(f"❌ Error sending Slack alert: {response.status_code} - {response.text}", file=sys.stderr)
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Error sending Slack alert: {e}", file=sys.stderr)
        return False

def main():
    """Main entry point."""

    if len(sys.argv) < 2:
        print("Usage: python send_slack_alert.py '<json_alert_data>'")
        print("\nExample:")
        print("""  python send_slack_alert.py '{
            "title": "Classifier Regression Detected",
            "current_accuracy": 0.853,
            "baseline_accuracy": 0.950,
            "drop_percentage": 0.097,
            "affected_complexity": ["complex", "research"],
            "severity": "CRITICAL",
            "timestamp": "2025-11-13T10:30:00"
        }'""")
        sys.exit(1)

    try:
        alert_data = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate required fields
    required_fields = ['title', 'current_accuracy', 'baseline_accuracy',
                      'drop_percentage', 'affected_complexity', 'severity', 'timestamp']
    missing = [f for f in required_fields if f not in alert_data]
    if missing:
        print(f"Error: Missing required fields: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    # Send alert
    success = send_alert(alert_data)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
