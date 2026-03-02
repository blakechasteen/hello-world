#!/usr/bin/env python3
"""
Domain-Specific AutoSpin Router
================================

Automatically detects domain from audio transcription and routes
to specialized spinners (BeeInspectionSpinner, RecipeSpinner, etc.)

Philosophy: "The transcription tells us what it is."

Author: Claude Code
Date: November 2025
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

from hololoom.spinningWheel.protocol import BaseSpinner, SpinResult
from hololoom.protocols.types import MemoryShard


# ============================================================================
# Domain Detection
# ============================================================================

# Domain detection patterns
DOMAIN_PATTERNS = {
    'bee_inspection': [
        r'\b(bee|hive|colony|brood|honey|pollen|queen|drone|frame|varroa|mite)\b',
        r'\b(inspection|check|examined|observed)\b',
        r'\b(temperature|weather|activity|population)\b'
    ],
    'recipe': [
        r'\b(recipe|ingredient|cup|tablespoon|teaspoon|ounce|gram)\b',
        r'\b(preheat|bake|cook|mix|stir|blend|chop|dice|simmer)\b',
        r'\b(oven|stove|pan|bowl|minutes|degrees)\b'
    ],
    'workout': [
        r'\b(reps|sets|workout|exercise|cardio|strength|training)\b',
        r'\b(pushup|squat|deadlift|bench|curl|pullup)\b',
        r'\b(weight|pounds|kilograms|lbs|kg)\b'
    ],
    'journal': [
        r'\b(today|yesterday|tomorrow|feeling|think|remember)\b',
        r'\b(grateful|mood|emotion|reflect)\b'
    ],
    'meeting_notes': [
        r'\b(meeting|agenda|action item|follow up|decision|discuss)\b',
        r'\b(attendee|participant|presenter)\b'
    ],
    'budget': [
        r'\b(spent|bought|dollar|cost|expense|budget)\b',
        r'\b(groceries|dining|transport|utilities)\b'
    ],
    'expense': [
        r'\b(receipt|vendor|invoice|reimbursable)\b',
        r'\b(expense|payment|transaction)\b'
    ],
    'time_tracking': [
        r'\b(worked|project|task|billable|hour)\b',
        r'\b(client|deadline|milestone)\b'
    ],
    'sop': [
        r'\b(procedure|protocol|step|process|guideline)\b',
        r'\b(safety|requirement|standard|policy)\b'
    ]
}


@dataclass
class DomainScore:
    """Score for domain classification"""
    domain: str
    confidence: float
    matched_patterns: List[str]


class DomainRouter:
    """
    Routes audio transcriptions to domain-specific spinners.

    Usage:
        router = DomainRouter()

        # Detect domain from transcription
        domain = router.detect_domain(transcription_text)

        # Route to specialized spinner
        shards = await router.route(transcription_text, domain)
    """

    def __init__(self, confidence_threshold: float = 0.3):
        """
        Initialize router.

        Args:
            confidence_threshold: Minimum confidence to classify as domain
        """
        self.confidence_threshold = confidence_threshold
        self.domain_spinners = {}  # Will be populated by register_spinner()

    def detect_domain(self, text: str, top_k: int = 3) -> List[DomainScore]:
        """
        Detect domain(s) from text.

        Args:
            text: Transcription text
            top_k: Return top K domains

        Returns:
            List of DomainScore objects, sorted by confidence
        """
        text_lower = text.lower()
        scores = []

        for domain, patterns in DOMAIN_PATTERNS.items():
            matches = []
            total_matches = 0

            for pattern in patterns:
                found = re.findall(pattern, text_lower)
                if found:
                    matches.append(pattern)
                    total_matches += len(found)

            # Score = (matched patterns / total patterns) * match_frequency
            if matches:
                pattern_coverage = len(matches) / len(patterns)
                match_density = min(total_matches / 10, 1.0)  # Normalize to 0-1
                confidence = (pattern_coverage * 0.7) + (match_density * 0.3)

                scores.append(DomainScore(
                    domain=domain,
                    confidence=confidence,
                    matched_patterns=matches
                ))

        # Sort by confidence and return top K
        scores.sort(key=lambda x: x.confidence, reverse=True)
        return scores[:top_k]

    def register_spinner(self, domain: str, spinner: BaseSpinner):
        """Register domain-specific spinner"""
        self.domain_spinners[domain] = spinner

    async def route(
        self,
        transcription_text: str,
        detected_domain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[MemoryShard]:
        """
        Route transcription to specialized spinner.

        Args:
            transcription_text: Full transcription text
            detected_domain: Pre-detected domain (or None to auto-detect)
            metadata: Additional metadata from transcription

        Returns:
            List of specialized MemoryShards
        """
        # Auto-detect if not provided
        if detected_domain is None:
            domain_scores = self.detect_domain(transcription_text)

            if not domain_scores or domain_scores[0].confidence < self.confidence_threshold:
                # No confident match - return generic shard
                return self._generic_spinner(transcription_text, metadata)

            detected_domain = domain_scores[0].domain

        # Route to specialized spinner
        spinner = self.domain_spinners.get(detected_domain)

        if spinner is None:
            print(f"[DomainRouter] No spinner registered for '{detected_domain}', using generic")
            return self._generic_spinner(transcription_text, metadata)

        # Process with specialized spinner
        print(f"[DomainRouter] Routing to '{detected_domain}' spinner")
        result = await spinner.spin({
            'text': transcription_text,
            'metadata': metadata or {}
        })

        return result.shards if hasattr(result, 'shards') else result

    def _generic_spinner(self, text: str, metadata: Optional[Dict] = None) -> List[MemoryShard]:
        """Fallback for unrecognized domains"""
        return [MemoryShard(
            id=f"generic_{hash(text[:100])}",
            text=text,
            episode="generic_transcription",
            entities=[],
            motifs=["transcription"],
            metadata=metadata or {}
        )]


# ============================================================================
# Domain-Specific Spinners
# ============================================================================

class BeeInspectionSpinner(BaseSpinner):
    """Specialized spinner for bee inspection notes"""

    def __init__(self):
        super().__init__(name="bee_inspection")

    async def _spin_impl(self, source: Dict[str, Any], **kwargs) -> List[MemoryShard]:
        """Process bee inspection transcription"""
        text = source['text']
        metadata = source.get('metadata', {})

        # Extract structured data
        extracted = self._extract_inspection_data(text)

        # Create specialized shard
        return [MemoryShard(
            id=f"bee_inspection_{metadata.get('timestamp', hash(text))}",
            text=text,
            episode="bee_inspection",
            entities=extracted['colonies'] + extracted['issues'],
            motifs=['beekeeping', 'inspection'] + extracted['observations'],
            metadata={
                **metadata,
                'domain': 'bee_inspection',
                'colonies': extracted['colonies'],
                'temperature': extracted.get('temperature'),
                'weather': extracted.get('weather'),
                'issues_found': extracted['issues'],
                'actions_needed': extracted['actions']
            }
        )]

    def _extract_inspection_data(self, text: str) -> Dict[str, Any]:
        """Extract structured data from bee inspection notes"""
        # Temperature extraction
        temp_match = re.search(r'(-?\d+)\s*(degrees?|°|F|C)', text, re.IGNORECASE)
        temperature = float(temp_match.group(1)) if temp_match else None

        # Colony names (hive IDs)
        colonies = re.findall(r'\b(hive|colony)\s+(\w+|\d+)', text, re.IGNORECASE)
        colonies = [c[1] for c in colonies]

        # Issues
        issues = []
        if re.search(r'\b(varroa|mite|pest)', text, re.IGNORECASE):
            issues.append('mite_concern')
        if re.search(r'\b(weak|struggling|failing)', text, re.IGNORECASE):
            issues.append('weak_colony')
        if re.search(r'\b(queen.*not|no.*queen)', text, re.IGNORECASE):
            issues.append('queenless')

        # Actions needed
        actions = []
        if re.search(r'\b(feed|sugar|syrup)', text, re.IGNORECASE):
            actions.append('feeding_required')
        if re.search(r'\b(treat|medication)', text, re.IGNORECASE):
            actions.append('treatment_needed')

        # Observations
        observations = []
        if re.search(r'\b(brood|larvae|eggs)', text, re.IGNORECASE):
            observations.append('brood_observed')
        if re.search(r'\b(honey|nectar)', text, re.IGNORECASE):
            observations.append('honey_stores')

        return {
            'temperature': temperature,
            'colonies': colonies,
            'issues': issues,
            'actions': actions,
            'observations': observations
        }


class RecipeSpinner(BaseSpinner):
    """Specialized spinner for recipe transcriptions"""

    def __init__(self):
        super().__init__(name="recipe")

    async def _spin_impl(self, source: Dict[str, Any], **kwargs) -> List[MemoryShard]:
        """Process recipe transcription"""
        text = source['text']
        metadata = source.get('metadata', {})

        # Extract structured recipe data
        extracted = self._extract_recipe_data(text)

        # Create specialized shard
        return [MemoryShard(
            id=f"recipe_{extracted['name'].replace(' ', '_').lower()}",
            text=text,
            episode="recipes",
            entities=extracted['ingredients'],
            motifs=['cooking', 'recipe'] + extracted['methods'],
            metadata={
                **metadata,
                'domain': 'recipe',
                'recipe_name': extracted['name'],
                'ingredients': extracted['ingredients'],
                'methods': extracted['methods'],
                'estimated_time': extracted.get('time'),
                'servings': extracted.get('servings')
            }
        )]

    def _extract_recipe_data(self, text: str) -> Dict[str, Any]:
        """Extract structured recipe data"""
        # Recipe name (often first sentence or after "recipe for")
        name_match = re.search(r'(?:recipe for|making|how to make)\s+([^.!?\n]+)', text, re.IGNORECASE)
        name = name_match.group(1).strip() if name_match else "Untitled Recipe"

        # Ingredients (words followed by measurements)
        ingredients = []
        ing_patterns = [
            r'(\d+(?:\.\d+)?)\s*(cup|tablespoon|teaspoon|ounce|gram|pound)s?\s+(?:of\s+)?(\w+)',
            r'(\w+)(?:,|\s+and\s+)',  # Simple ingredient lists
        ]
        for pattern in ing_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                ingredients.extend([m[-1] for m in matches])

        # Cooking methods
        methods = []
        method_keywords = ['bake', 'cook', 'fry', 'boil', 'grill', 'roast', 'simmer', 'sauté']
        for method in method_keywords:
            if re.search(rf'\b{method}', text, re.IGNORECASE):
                methods.append(method)

        # Time estimation
        time_match = re.search(r'(\d+)\s*(minute|hour)s?', text, re.IGNORECASE)
        time = f"{time_match.group(1)} {time_match.group(2)}s" if time_match else None

        # Servings
        servings_match = re.search(r'(?:serves?|servings?:?)\s*(\d+)', text, re.IGNORECASE)
        servings = int(servings_match.group(1)) if servings_match else None

        return {
            'name': name,
            'ingredients': list(set(ingredients)),  # Deduplicate
            'methods': methods,
            'time': time,
            'servings': servings
        }


class BudgetSpinner(BaseSpinner):
    """Specialized spinner for budget/expense entries"""

    def __init__(self):
        super().__init__(name="budget")

    async def _spin_impl(self, source: Dict[str, Any], **kwargs) -> List[MemoryShard]:
        """Process budget/expense transcription"""
        text = source['text']
        metadata = source.get('metadata', {})

        extracted = self._extract_budget_data(text)

        return [MemoryShard(
            id=f"budget_{hash(text[:100])}",
            text=text,
            episode="budget",
            entities=[extracted.get('category', 'uncategorized')],
            motifs=['expense', 'budget', extracted.get('category', 'other')],
            metadata={
                **metadata,
                'domain': 'budget',
                **extracted
            }
        )]

    def _extract_budget_data(self, text: str) -> Dict[str, Any]:
        """Extract budget/expense data"""
        data = {}

        # Amount
        amount_patterns = [
            r'\$(\d+(?:\.\d{2})?)',
            r'(\d+(?:\.\d{2})?)\s*dollars?',
        ]
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['amount'] = float(match.group(1))
                break

        # Category
        categories = {
            'groceries': r'\b(groceries|grocery|food)\b',
            'dining': r'\b(restaurant|dining|eating out)\b',
            'transport': r'\b(gas|fuel|uber|taxi)\b',
        }
        for cat, pattern in categories.items():
            if re.search(pattern, text, re.IGNORECASE):
                data['category'] = cat
                break

        return data
