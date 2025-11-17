#!/usr/bin/env python3
"""
Email Newsletter Digest Workflow Template
==========================================

**Impact**: Save 1 hour/week reading newsletters
**Success Rate**: 90%
**Setup Time**: 2 minutes

Aggregates email newsletters, extracts key insights,
and generates a weekly digest with highlights.

**Workflow**:
1. Newsletter Fetcher → Fetch newsletters from inbox (last 7 days)
2. Content Extractor → Extract main content, links, images
3. Synthesizer → Aggregate and summarize key insights
4. Response Generator → Format as weekly digest (Markdown)

**Integrations**: Gmail API
**Created**: November 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
from datetime import datetime


@dataclass
class NewsletterDigestTemplate:
    """Email Newsletter Digest Workflow Template"""

    id: str = "newsletter-digest"
    name: str = "Email Newsletter Digest"
    category: str = "Email & Communication"
    difficulty: str = "Beginner"
    version: str = "1.0.0"
    estimated_time_saved: str = "1 hour/week"
    success_rate: float = 0.90
    setup_time_minutes: int = 2
    tags: List[str] = field(default_factory=lambda: [
        "newsletters", "digest", "summarization", "content"
    ])

    def get_workflow_definition(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'name': self.name,
            'description': 'Aggregate newsletters into weekly digest',
            'nodes': [
                {
                    'id': 'newsletter_fetcher',
                    'agentType': 'newsletter_fetcher',
                    'x': 100,
                    'y': 200,
                    'config': {
                        'identification_method': 'auto',
                        'date_range': '7d',
                        'max_newsletters': 20
                    }
                },
                {
                    'id': 'content_extractor',
                    'agentType': 'content_extractor',
                    'x': 300,
                    'y': 200,
                    'config': {
                        'extract_links': True,
                        'extract_images': False,
                        'summarize': True,
                        'max_summary_length': 3
                    }
                },
                {
                    'id': 'synthesizer',
                    'agentType': 'synthesizer',
                    'x': 500,
                    'y': 200,
                    'config': {
                        'extract_entities': True,
                        'extract_motifs': True
                    }
                },
                {
                    'id': 'formatter',
                    'agentType': 'response',
                    'x': 700,
                    'y': 200,
                    'config': {
                        'format': 'markdown'
                    }
                }
            ],
            'connections': [
                {'id': 'conn_1', 'from': 'newsletter_fetcher', 'to': 'content_extractor'},
                {'id': 'conn_2', 'from': 'content_extractor', 'to': 'synthesizer'},
                {'id': 'conn_3', 'from': 'synthesizer', 'to': 'formatter'}
            ],
            'metadata': {
                'template_id': self.id,
                'category': self.category,
                'tags': self.tags
            }
        }

    def get_required_credentials(self) -> List[str]:
        return ['GMAIL_API_KEY']

    def get_test_data(self) -> Dict[str, Any]:
        return {
            'sample_newsletters': [
                {
                    'from': 'TechCrunch',
                    'subject': 'Daily Crunch: AI startup raises $100M',
                    'key_insights': ['AI funding boom', 'New regulations proposed']
                },
                {
                    'from': 'Morning Brew',
                    'subject': 'Tesla stock surges',
                    'key_insights': ['EV market growth', 'Competition intensifies']
                }
            ]
        }

    def estimate_impact(self, newsletters_per_week: int = 20) -> Dict[str, Any]:
        manual_minutes_per_newsletter = 3
        automated_minutes_total = 5

        manual_total = newsletters_per_week * manual_minutes_per_newsletter
        time_saved = (manual_total - automated_minutes_total) / 60

        return {
            'newsletters_per_week': newsletters_per_week,
            'time_saved_per_week': f'{time_saved:.1f} hours',
            'yearly_time_saved': f'{time_saved * 50:.0f} hours',
            'yearly_savings': f'${time_saved * 50 * 100:.2f}'
        }
