"""
Domain Houses - Pre-configured WeaveHouse Implementations
=========================================================

Ready-to-use WeaveHouse configurations for common use cases.

Available Domain Houses:
- CodeReviewHouse: Multi-perspective code review
- ResearchHouse: Deep research with verification

Each house combines the core Looms (RECALL, REASON, REACH, REFLECT, REFUSE)
with domain-specific configurations and consensus strategies.

Date: December 2025
Author: HoloLoom Architecture Team
"""

from .code_review_house import (
    CodeReviewHouse,
    create_code_review_house,
)

from .research_house import (
    ResearchHouse,
    create_research_house,
)

__all__ = [
    # Code Review
    "CodeReviewHouse",
    "create_code_review_house",

    # Research
    "ResearchHouse",
    "create_research_house",
]
