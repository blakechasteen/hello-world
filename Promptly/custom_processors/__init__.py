#!/usr/bin/env python3
"""
Custom Chain Processors for Workflow Extension

This package provides advanced processors for extending Promptly workflows:
- WebhookProcessor: Send HTTP webhooks during chain execution
- APIIntegrationProcessor: Call external APIs with authentication
- DataValidationProcessor: Schema validation and data quality checks
- CacheProcessor: Multi-level caching with metrics
- AggregationProcessor: Aggregate results from parallel branches
"""

from .webhook import WebhookProcessor
from .api import APIIntegrationProcessor
from .validation import DataValidationProcessor
from .cache import CacheProcessor
from .aggregation import AggregationProcessor

__all__ = [
    "WebhookProcessor",
    "APIIntegrationProcessor",
    "DataValidationProcessor",
    "CacheProcessor",
    "AggregationProcessor"
]

__version__ = "1.0.0"
