"""
Web Skills Package
==================
Skills for web automation, scraping, and API interactions.

Available Skills:
- playwright: Browser automation and testing
- rest_api_client: Generic HTTP/REST API operations
"""

from .playwright_skill import PlaywrightSkill
from .rest_api_client import RESTAPIClientSkill

__all__ = ['PlaywrightSkill', 'RESTAPIClientSkill']
