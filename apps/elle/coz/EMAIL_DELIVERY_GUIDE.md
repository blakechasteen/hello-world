# COZ Email Delivery System

Status: Production Ready
Created: 2025-11-22

## Quick Start

1. Install markdown: pip install markdown
2. Copy email_config.example.json to email_config.json and configure
3. Copy recipients.example.json to recipients.json and add emails
4. Run: python send_daily_brief.py

## Usage

Send daily brief via command line:
    python send_daily_brief.py

Programmatic API:
    from elle.coz.email_delivery import EmailDelivery, EmailConfig
    config = EmailConfig.gmail('user@gmail.com', 'app_password')
    delivery = EmailDelivery(config)
    delivery.send(recipients, brief)

## Configuration

email_config.json:
    Gmail: Use app password from myaccount.google.com/apppasswords
    SendGrid: Use API key with smtp.sendgrid.net

recipients.json:
    Groups: managers, team_leads, executives, all
    Add emails to appropriate groups

## Automation

Daily cron job (9 AM):
    0 9 * * * cd /path/to/elle/coz && python send_daily_brief.py

## Files

- email_delivery.py: Core system
- send_daily_brief.py: CLI script
- email_config.example.json: Example config
- recipients.example.json: Example recipients
