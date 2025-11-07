"""
Email Template
==============

Templates for email generation.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base import (
    TemplateProtocol,
    TemplateType,
    TemplateContext,
    TemplateResult
)

logger = logging.getLogger(__name__)


class EmailTemplate:
    """
    Email template with customizable tone and structure.

    Supports:
    - Professional, casual, follow-up styles
    - Standard email structure (greeting, body, closing, signature)
    - Context integration from memories
    """

    def __init__(
        self,
        style: str = 'professional',  # professional, casual, follow_up
        structure: str = 'standard'     # standard, brief, detailed
    ):
        """
        Initialize email template.

        Args:
            style: Email style (professional, casual, follow_up)
            structure: Email structure (standard, brief, detailed)
        """
        self.style = style
        self.structure = structure
        self.logger = logger

    async def generate(self, context: TemplateContext) -> TemplateResult:
        """
        Generate email from template context.

        Args:
            context: Template context with variables and memories

        Returns:
            Template result with email content
        """
        variables = context.variables
        memories = context.memories

        # Extract required variables
        recipient = variables.get('recipient', 'there')
        subject = variables.get('subject', 'Update')
        purpose = variables.get('purpose', 'general')

        # Build email sections
        sections = {}

        # 1. Subject line
        sections['subject'] = self._generate_subject(subject, purpose, variables)

        # 2. Greeting
        sections['greeting'] = self._generate_greeting(recipient, self.style)

        # 3. Opening
        sections['opening'] = self._generate_opening(purpose, memories, self.style)

        # 4. Body
        sections['body'] = self._generate_body(purpose, memories, self.structure)

        # 5. Closing
        sections['closing'] = self._generate_closing(purpose, self.style)

        # 6. Signature
        sender = variables.get('sender', '')
        sections['signature'] = self._generate_signature(sender, self.style)

        # Assemble email
        email_parts = [
            f"Subject: {sections['subject']}",
            "",  # Blank line
            sections['greeting'],
            "",
            sections['opening'],
            "",
            sections['body'],
            "",
            sections['closing'],
            sections['signature']
        ]

        content = '\n'.join(email_parts)

        return TemplateResult(
            content=content,
            template_type=TemplateType.EMAIL,
            variables_used=list(variables.keys()),
            sections=sections,
            metadata={
                'style': self.style,
                'structure': self.structure,
                'word_count': len(content.split())
            }
        )

    def _generate_subject(
        self,
        subject: str,
        purpose: str,
        variables: Dict[str, Any]
    ) -> str:
        """Generate subject line."""
        # If explicit subject provided, use it
        if subject and subject != 'Update':
            return subject

        # Generate based on purpose
        purpose_subjects = {
            'project_update': 'Project Update',
            'request': 'Request for Information',
            'follow_up': 'Following Up',
            'introduction': 'Introduction',
            'thank_you': 'Thank You',
            'meeting': 'Meeting Request'
        }

        return purpose_subjects.get(purpose, 'Update')

    def _generate_greeting(self, recipient: str, style: str) -> str:
        """Generate greeting."""
        if style == 'professional':
            if ',' in recipient or ' ' in recipient:
                # Full name or multiple recipients
                return f"Dear {recipient},"
            else:
                # Single name
                return f"Dear {recipient},"
        elif style == 'casual':
            return f"Hi {recipient},"
        elif style == 'follow_up':
            return f"Hi {recipient},"
        else:
            return f"Hello {recipient},"

    def _generate_opening(
        self,
        purpose: str,
        memories: List,
        style: str
    ) -> str:
        """Generate opening line."""
        if style == 'professional':
            openings = {
                'project_update': 'I hope this email finds you well. I wanted to provide an update on our project.',
                'request': 'I hope this message finds you well. I am writing to request information about',
                'follow_up': 'I wanted to follow up on our previous conversation regarding',
                'introduction': 'I hope this email finds you well. I wanted to introduce myself and',
                'thank_you': 'I wanted to take a moment to thank you for',
                'meeting': 'I hope this email finds you well. I would like to schedule a meeting to discuss'
            }
        elif style == 'casual':
            openings = {
                'project_update': 'Quick update on the project!',
                'request': 'I wanted to ask you about',
                'follow_up': 'Just following up on',
                'introduction': 'I wanted to reach out and introduce myself!',
                'thank_you': 'Thanks so much for',
                'meeting': 'Can we schedule some time to chat about'
            }
        else:
            openings = {
                'project_update': 'Following up with an update:',
                'request': 'I wanted to check in about',
                'follow_up': 'Circling back on',
                'introduction': 'Reaching out to connect!',
                'thank_you': 'I appreciate',
                'meeting': 'Could we set up time to discuss'
            }

        base_opening = openings.get(purpose, 'I wanted to reach out about')

        # Add context from first memory if available
        if memories and len(memories) > 0:
            first_memory = memories[0].text
            # Take first sentence or first 50 words
            context = ' '.join(first_memory.split()[:50])
            if purpose not in ['thank_you', 'introduction']:
                base_opening += f' {context.lower()}'

        return base_opening

    def _generate_body(
        self,
        purpose: str,
        memories: List,
        structure: str
    ) -> str:
        """Generate email body."""
        if structure == 'brief':
            # Single paragraph from top memory
            if memories:
                return memories[0].text
            return "Please let me know if you have any questions."

        elif structure == 'detailed':
            # Multiple paragraphs from memories
            paragraphs = []
            for memory in memories[:4]:  # Up to 4 paragraphs
                paragraphs.append(memory.text)
            return '\n\n'.join(paragraphs)

        else:  # standard
            # 2-3 paragraphs
            paragraphs = []

            # Main content from memories
            for memory in memories[:2]:
                paragraphs.append(memory.text)

            # Add transition if we have 2+ paragraphs
            if len(paragraphs) >= 2:
                return '\n\n'.join(paragraphs)
            elif paragraphs:
                return paragraphs[0]
            else:
                return "I wanted to share this information with you."

    def _generate_closing(self, purpose: str, style: str) -> str:
        """Generate closing."""
        if style == 'professional':
            closings = {
                'project_update': 'Please let me know if you have any questions or need additional information.',
                'request': 'I would greatly appreciate your assistance with this matter.',
                'follow_up': 'I look forward to your response.',
                'introduction': 'I look forward to connecting with you.',
                'thank_you': 'Thank you again for your time and support.',
                'meeting': 'Please let me know your availability.'
            }
            closing = closings.get(purpose, 'Please let me know if you have any questions.')
            return closing

        elif style == 'casual':
            closings = {
                'project_update': 'Let me know if you have questions!',
                'request': 'Thanks in advance!',
                'follow_up': 'Looking forward to hearing from you!',
                'introduction': 'Excited to connect!',
                'thank_you': 'Thanks again!',
                'meeting': 'Let me know what works for you!'
            }
            return closings.get(purpose, 'Let me know if you have questions!')

        else:  # follow_up
            return 'Looking forward to your response.'

    def _generate_signature(self, sender: str, style: str) -> str:
        """Generate signature."""
        if not sender:
            sender = 'Best'

        if style == 'professional':
            return f"Best regards,\n{sender}"
        elif style == 'casual':
            return f"Thanks,\n{sender}"
        else:
            return f"Best,\n{sender}"

    def get_required_variables(self) -> List[str]:
        """Get required template variables."""
        return ['recipient']

    def get_optional_variables(self) -> List[str]:
        """Get optional template variables."""
        return ['subject', 'purpose', 'sender']

    @property
    def template_type(self) -> TemplateType:
        """Template type."""
        return TemplateType.EMAIL
