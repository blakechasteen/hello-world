# COZ Email Delivery System
# Created: 2025-11-22

import smtplib
import json
import logging
from pathlib import Path
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    print("Warning: markdown not installed. Run: pip install markdown")

class EmailConfig:
    def __init__(self, smtp_host='smtp.gmail.com', smtp_port=587, 
                 smtp_user='', smtp_password='', from_email='', 
                 from_name='COZ Intelligence', use_tls=True):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_email = from_email or smtp_user
        self.from_name = from_name
        self.use_tls = use_tls
    
    @classmethod
    def gmail(cls, email, password):
        return cls(smtp_user=email, smtp_password=password, from_email=email)

class EmailDelivery:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def markdown_to_html(self, text):
        if MARKDOWN_AVAILABLE:
            return markdown.markdown(text, extensions=['tables'])
        return text.replace('
', '<br>')
    
    def build_html(self, brief):
        perf = brief.get('performance_indicators', {})
        html = f'''<html><body style="font-family: sans-serif; max-width: 700px; margin: 0 auto;">
        <div style="background: #667eea; color: white; padding: 30px; text-align: center;">
            <h1>COZ Daily Brief</h1>
            <div>{brief.get('date', '')}</div>
        </div>
        <div style="padding: 40px;">
            <h2>Metrics</h2>
            <p>Profit: {perf.get('profit_margin', 0):.1f}%</p>
            <p>Efficiency: {perf.get('task_efficiency', 0)*100:.1f}%</p>
            {self.markdown_to_html(brief.get('summary', ''))}
        </div>
        </body></html>'''
        return html
    
    def send(self, recipients, brief):
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = f'{self.config.from_name} <{self.config.from_email}>'
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"COZ Daily Brief - {brief.get('date', '')}"
            
            msg.attach(MIMEText(brief.get('summary', ''), 'plain'))
            msg.attach(MIMEText(self.build_html(brief), 'html'))
            
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                if self.config.use_tls:
                    server.starttls()
                if self.config.smtp_user and self.config.smtp_password:
                    server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)
                self.logger.info(f'Email sent to {len(recipients)} recipients')
                return True
        except Exception as e:
            self.logger.error(f'Email failed: {e}')
            return False

class RecipientManager:
    def __init__(self, file='recipients.json'):
        self.file = file
        self.data = self.load()
    
    def load(self):
        if Path(self.file).exists():
            with open(self.file) as f:
                return json.load(f)
        default = {'managers': [], 'team_leads': [], 'all': []}
        self.save(default)
        return default
    
    def save(self, data):
        with open(self.file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add(self, email, group='all'):
        if group not in self.data:
            self.data[group] = []
        if email not in self.data[group]:
            self.data[group].append(email)
            self.save(self.data)
    
    def get(self, *groups):
        emails = set()
        for group in groups:
            if group in self.data:
                emails.update(self.data[group])
        return list(emails)

if __name__ == '__main__':
    print('COZ Email Delivery System')
    print('Usage: from elle.coz.email_delivery import EmailDelivery, EmailConfig')
