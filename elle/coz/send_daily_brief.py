#!/usr/bin/env python3
# Send COZ Daily Brief via Email
# Created: 2025-11-22

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from elle.coz.sync_manager import SyncManager
from elle.coz.intelligence import IntelligenceEngine
from elle.coz.email_delivery import EmailDelivery, EmailConfig, RecipientManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Parse all data
        logger.info('Parsing COZ data...')
        sync = SyncManager()
        sync.parse_all()
        
        # Generate daily brief
        logger.info('Generating daily brief...')
        intelligence = IntelligenceEngine(sync, logger=logger)
        brief = intelligence.generate_daily_brief(
            hourly_rate=25.0,
            use_refinement=True,  # Use HoloLoom refinement
            refinement_provider='anthropic'
        )
        
        logger.info(f"Brief generated: {brief['date']}")
        logger.info(f"Refinement used: {brief['refinement_used']}")
        
        # Load email config
        try:
            config = EmailConfig.from_file('email_config.json')
        except FileNotFoundError:
            logger.error('email_config.json not found. Please create it first.')
            logger.info('Example: EmailConfig.gmail("user@gmail.com", "app_password")')
            return 1
        
        # Load recipients
        recipients_mgr = RecipientManager()
        recipients = recipients_mgr.get('managers', 'executives', 'all')
        
        if not recipients:
            logger.error('No recipients configured. Add recipients first.')
            logger.info('Example: recipients_mgr.add("manager@example.com", "managers")')
            return 1
        
        logger.info(f'Sending to {len(recipients)} recipients: {recipients}')
        
        # Send email
        delivery = EmailDelivery(config)
        success = delivery.send(recipients, brief)
        
        if success:
            logger.info('Daily brief sent successfully!')
            return 0
        else:
            logger.error('Failed to send daily brief')
            return 1
    
    except Exception as e:
        logger.error(f'Error: {e}', exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
