#!/usr/bin/env python3
"""
Send email alert for HoloLoom Phase 3 regression.

Usage:
    python send_email_alert.py '{"title": "...", "current_accuracy": 0.85, ...}'

Environment Variables:
    SMTP_HOST - SMTP server (default: smtp.gmail.com)
    SMTP_PORT - SMTP port (default: 587)
    SMTP_USER - SMTP username (required)
    SMTP_PASSWORD - SMTP password (required)
    SMTP_FROM - From email address (default: SMTP_USER)
    SMTP_TO - Comma-separated list of recipient emails (required)
"""

import os
import sys
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any

# SMTP Configuration
SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
SMTP_FROM = os.getenv('SMTP_FROM', SMTP_USER)
SMTP_TO = os.getenv('SMTP_TO', '').split(',')

def format_plain_text(alert_data: Dict[str, Any]) -> str:
    """Format alert as plain text email."""

    severity_symbol = "🚨" if alert_data['severity'] == "CRITICAL" else "⚠️"

    text = f"""
{severity_symbol} {alert_data['title']}
{'=' * len(alert_data['title'])}

Current accuracy: {alert_data['current_accuracy']:.1%}
Baseline accuracy: {alert_data['baseline_accuracy']:.1%}
Drop: {alert_data['drop_percentage']:.1%} (threshold: 2.0%)

Affected complexities:
{chr(10).join(f"  • {c}" for c in alert_data['affected_complexity'])}

Time: {alert_data['timestamp']}
Severity: {alert_data['severity']}

Action Required:
{'-' * 40}
Please investigate the regression immediately:
  1. Review Grafana dashboard: http://localhost:3000/d/hololoom-phase3
  2. Check Prometheus metrics: http://localhost:9090
  3. Review recent pattern deployments
  4. Consider rollback if necessary

---
HoloLoom Phase 3 Adaptive Learning System
Automated Alert
"""
    return text

def format_html(alert_data: Dict[str, Any]) -> str:
    """Format alert as HTML email."""

    severity_emoji = "🚨" if alert_data['severity'] == "CRITICAL" else "⚠️"
    severity_color = "#d32f2f" if alert_data['severity'] == "CRITICAL" else "#ff9800"

    html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 600px;
    margin: 0 auto;
    padding: 20px;
}}
.header {{
    background-color: {severity_color};
    color: white;
    padding: 30px 20px;
    border-radius: 8px 8px 0 0;
    text-align: center;
}}
.header h1 {{
    margin: 0;
    font-size: 24px;
}}
.content {{
    background-color: #f9f9f9;
    padding: 30px 20px;
    border-left: 1px solid #ddd;
    border-right: 1px solid #ddd;
}}
.metric {{
    background-color: white;
    padding: 15px;
    margin: 15px 0;
    border-left: 4px solid #2196f3;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}
.metric strong {{
    display: block;
    color: #666;
    font-size: 12px;
    text-transform: uppercase;
    margin-bottom: 5px;
}}
.metric .value {{
    font-size: 24px;
    font-weight: bold;
    color: #333;
}}
.affected {{
    background-color: #fff3cd;
    border-left-color: #ffc107;
    padding: 15px;
    margin: 15px 0;
    border-left: 4px solid #ffc107;
    border-radius: 4px;
}}
.action-box {{
    background-color: white;
    padding: 20px;
    margin: 20px 0;
    border-left: 4px solid {severity_color};
    border-radius: 4px;
}}
.action-box h3 {{
    margin-top: 0;
    color: {severity_color};
}}
.action-box ol {{
    margin: 10px 0;
    padding-left: 20px;
}}
.action-box li {{
    margin: 10px 0;
}}
.button {{
    display: inline-block;
    padding: 12px 24px;
    margin: 10px 10px 10px 0;
    background-color: #2196f3;
    color: white;
    text-decoration: none;
    border-radius: 4px;
    font-weight: bold;
}}
.button:hover {{
    background-color: #1976d2;
}}
.footer {{
    background-color: #f5f5f5;
    padding: 20px;
    text-align: center;
    font-size: 12px;
    color: #666;
    border-radius: 0 0 8px 8px;
    border: 1px solid #ddd;
}}
</style>
</head>
<body>
<div class="header">
<h1>{severity_emoji} {alert_data['title']}</h1>
</div>

<div class="content">
<div class="metric">
<strong>Current Accuracy</strong>
<div class="value">{alert_data['current_accuracy']:.1%}</div>
</div>

<div class="metric">
<strong>Baseline Accuracy</strong>
<div class="value">{alert_data['baseline_accuracy']:.1%}</div>
</div>

<div class="metric">
<strong>Accuracy Drop</strong>
<div class="value" style="color: {severity_color};">{alert_data['drop_percentage']:.1%}</div>
<span style="color: #666; font-size: 14px;">Threshold: 2.0%</span>
</div>

<div class="affected">
<strong>Affected Complexities</strong><br>
{'<br>'.join(f"• <strong>{c}</strong>" for c in alert_data['affected_complexity'])}
</div>

<div class="metric">
<strong>Time</strong>
<div style="font-size: 16px; color: #666;">{alert_data['timestamp']}</div>
</div>

<div class="metric">
<strong>Severity</strong>
<div class="value" style="color: {severity_color};">{alert_data['severity']}</div>
</div>

<div class="action-box">
<h3>⚡ Action Required</h3>
<p>Please investigate this regression immediately:</p>
<ol>
<li>Review the <a href="http://localhost:3000/d/hololoom-phase3">Grafana dashboard</a></li>
<li>Check <a href="http://localhost:9090">Prometheus metrics</a></li>
<li>Review recent pattern deployments</li>
<li>Consider rollback if necessary</li>
</ol>

<div style="margin-top: 20px;">
<a href="http://localhost:3000/d/hololoom-phase3" class="button">View Dashboard</a>
<a href="http://localhost:9090" class="button" style="background-color: #ff9800;">View Metrics</a>
</div>
</div>
</div>

<div class="footer">
<strong>HoloLoom Phase 3 Adaptive Learning System</strong><br>
Automated Alert System<br>
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
</div>
</body>
</html>
"""
    return html

def send_email(alert_data: Dict[str, Any]) -> bool:
    """Send formatted email alert."""

    if not SMTP_USER or not SMTP_PASSWORD:
        print("Error: SMTP_USER and SMTP_PASSWORD environment variables must be set", file=sys.stderr)
        return False

    if not SMTP_TO or SMTP_TO == ['']:
        print("Error: SMTP_TO environment variable must be set with recipient emails", file=sys.stderr)
        return False

    # Create message
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"[{alert_data['severity']}] {alert_data['title']}"
    msg['From'] = SMTP_FROM
    msg['To'] = ', '.join(SMTP_TO)
    msg['X-Priority'] = '1' if alert_data['severity'] == 'CRITICAL' else '3'

    # Attach parts
    text_part = MIMEText(format_plain_text(alert_data), 'plain', 'utf-8')
    html_part = MIMEText(format_html(alert_data), 'html', 'utf-8')
    msg.attach(text_part)
    msg.attach(html_part)

    # Send email
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)

        print(f"✅ Email alert sent successfully to {', '.join(SMTP_TO)}")
        return True

    except smtplib.SMTPException as e:
        print(f"❌ SMTP error sending email: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"❌ Error sending email: {e}", file=sys.stderr)
        return False

def main():
    """Main entry point."""

    if len(sys.argv) < 2:
        print("Usage: python send_email_alert.py '<json_alert_data>'")
        print("\nExample:")
        print("""  python send_email_alert.py '{
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

    # Send email
    success = send_email(alert_data)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
