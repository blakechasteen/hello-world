#!/usr/bin/env python3
"""
Live Transcription Scratchpad
==============================

Real-time audio → structured templates with live editing.

Features:
- Streaming Whisper transcription
- Domain-specific templates (recipe, budget, time tracking, SOP, expenses, bee inspections)
- Live editing in web UI
- Auto-save to memory when finalized
- Template validation and suggestions

Author: Claude Code
Date: November 2025
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import re

from HoloLoom.Documentation.types import MemoryShard


# ============================================================================
# Template System
# ============================================================================

@dataclass
class TemplateField:
    """Single field in a template"""
    name: str
    field_type: str  # 'text', 'number', 'list', 'currency', 'duration'
    default: Any = None
    required: bool = False
    validation_pattern: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)


@dataclass
class DomainTemplate:
    """Domain-specific template structure"""
    name: str
    fields: List[TemplateField]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Export as editable dictionary"""
        return {
            field.name: field.default if field.default is not None else ""
            for field in self.fields
        }

    def validate(self, data: Dict[str, Any]) -> List[str]:
        """Validate filled template, return list of errors"""
        errors = []

        for field in self.fields:
            value = data.get(field.name)

            # Required check
            if field.required and not value:
                errors.append(f"'{field.name}' is required")
                continue

            # Type validation
            if value and field.validation_pattern:
                if not re.match(field.validation_pattern, str(value)):
                    errors.append(f"'{field.name}' has invalid format")

        return errors


# ============================================================================
# Domain Templates
# ============================================================================

RECIPE_TEMPLATE = DomainTemplate(
    name="recipe",
    fields=[
        TemplateField("recipe_name", "text", required=True),
        TemplateField("servings", "number", default=4),
        TemplateField("prep_time", "duration", suggestions=["15 min", "30 min", "1 hour"]),
        TemplateField("cook_time", "duration", suggestions=["15 min", "30 min", "45 min"]),
        TemplateField("ingredients", "list", default=[], required=True),
        TemplateField("instructions", "list", default=[], required=True),
        TemplateField("notes", "text"),
        TemplateField("tags", "list", suggestions=["breakfast", "lunch", "dinner", "dessert", "quick", "healthy"])
    ]
)

BUDGET_TEMPLATE = DomainTemplate(
    name="budget",
    fields=[
        TemplateField("date", "text", default=datetime.now().strftime("%Y-%m-%d"), required=True),
        TemplateField("category", "text", required=True,
                     suggestions=["groceries", "dining", "transport", "utilities", "entertainment", "healthcare", "other"]),
        TemplateField("amount", "currency", required=True, validation_pattern=r"^\d+(\.\d{2})?$"),
        TemplateField("description", "text", required=True),
        TemplateField("payment_method", "text", suggestions=["cash", "credit", "debit", "check"]),
        TemplateField("merchant", "text"),
        TemplateField("notes", "text"),
        TemplateField("recurring", "text", suggestions=["no", "weekly", "monthly", "yearly"])
    ]
)

TIME_TRACKING_TEMPLATE = DomainTemplate(
    name="time_tracking",
    fields=[
        TemplateField("date", "text", default=datetime.now().strftime("%Y-%m-%d"), required=True),
        TemplateField("project", "text", required=True),
        TemplateField("task", "text", required=True),
        TemplateField("duration", "duration", required=True, validation_pattern=r"^\d+(\.\d+)?\s*(min|hour|hr)s?$"),
        TemplateField("start_time", "text", validation_pattern=r"^\d{1,2}:\d{2}$"),
        TemplateField("end_time", "text", validation_pattern=r"^\d{1,2}:\d{2}$"),
        TemplateField("billable", "text", default="yes", suggestions=["yes", "no"]),
        TemplateField("notes", "text"),
        TemplateField("tags", "list", suggestions=["meeting", "coding", "research", "admin", "testing"])
    ]
)

EXPENSE_TEMPLATE = DomainTemplate(
    name="expense",
    fields=[
        TemplateField("date", "text", default=datetime.now().strftime("%Y-%m-%d"), required=True),
        TemplateField("vendor", "text", required=True),
        TemplateField("amount", "currency", required=True, validation_pattern=r"^\d+(\.\d{2})?$"),
        TemplateField("category", "text", required=True,
                     suggestions=["office supplies", "travel", "meals", "equipment", "software", "services"]),
        TemplateField("description", "text", required=True),
        TemplateField("receipt_number", "text"),
        TemplateField("reimbursable", "text", default="yes", suggestions=["yes", "no"]),
        TemplateField("project", "text"),
        TemplateField("notes", "text")
    ]
)

BEE_INSPECTION_TEMPLATE = DomainTemplate(
    name="bee_inspection",
    fields=[
        TemplateField("date", "text", default=datetime.now().strftime("%Y-%m-%d"), required=True),
        TemplateField("hive_id", "text", required=True),
        TemplateField("temperature", "number", validation_pattern=r"^-?\d+(\.\d+)?$"),
        TemplateField("weather", "text", suggestions=["sunny", "cloudy", "rainy", "windy"]),
        TemplateField("queen_seen", "text", suggestions=["yes", "no", "unsure"]),
        TemplateField("brood_pattern", "text", suggestions=["excellent", "good", "spotty", "poor", "none"]),
        TemplateField("honey_stores", "text", suggestions=["full", "adequate", "low", "empty"]),
        TemplateField("pests_observed", "list", suggestions=["varroa", "hive beetle", "wax moth", "none"]),
        TemplateField("issues", "list"),
        TemplateField("actions_needed", "list", suggestions=["feed", "treat", "requeen", "monitor"]),
        TemplateField("notes", "text")
    ]
)

SOP_TEMPLATE = DomainTemplate(
    name="sop",
    fields=[
        TemplateField("title", "text", required=True),
        TemplateField("purpose", "text", required=True),
        TemplateField("scope", "text", suggestions=["all staff", "managers only", "specific department"]),
        TemplateField("responsibilities", "list", required=True),
        TemplateField("procedure_steps", "list", required=True),
        TemplateField("required_materials", "list"),
        TemplateField("safety_warnings", "list"),
        TemplateField("frequency", "text", suggestions=["daily", "weekly", "monthly", "as needed", "seasonal"]),
        TemplateField("estimated_duration", "duration"),
        TemplateField("related_sops", "list"),
        TemplateField("revision_date", "text", default=datetime.now().strftime("%Y-%m-%d")),
        TemplateField("approved_by", "text"),
        TemplateField("notes", "text")
    ]
)


# ============================================================================
# Live Scratchpad
# ============================================================================

class LiveScratchpad:
    """
    Real-time transcription → template population.

    Usage:
        scratchpad = LiveScratchpad()

        # Start recording
        await scratchpad.start_recording("recipe")

        # Transcription populates template in real-time
        # User can edit in web UI

        # Finalize and save
        shard = await scratchpad.finalize()
    """

    def __init__(self):
        # Template registry
        self.templates = {
            'recipe': RECIPE_TEMPLATE,
            'budget': BUDGET_TEMPLATE,
            'time_tracking': TIME_TRACKING_TEMPLATE,
            'expense': EXPENSE_TEMPLATE,
            'bee_inspection': BEE_INSPECTION_TEMPLATE,
            'sop': SOP_TEMPLATE,
        }

        # Active session
        self.active_domain: Optional[str] = None
        self.active_template: Optional[DomainTemplate] = None
        self.current_data: Dict[str, Any] = {}
        self.transcription_buffer: str = ""

        # Callbacks for UI updates
        self.on_update_callbacks: List[Callable] = []

    def register_update_callback(self, callback: Callable):
        """Register callback for real-time UI updates"""
        self.on_update_callbacks.append(callback)

    def _notify_update(self):
        """Notify all registered callbacks"""
        for callback in self.on_update_callbacks:
            try:
                callback(self.current_data)
            except Exception as e:
                print(f"[LiveScratchpad] Callback error: {e}")

    async def start_recording(
        self,
        domain: str,
        auto_detect: bool = False
    ):
        """
        Start live recording session.

        Args:
            domain: Target domain ('recipe', 'budget', etc.) or None for auto-detect
            auto_detect: Auto-detect domain from speech
        """
        if not auto_detect:
            if domain not in self.templates:
                raise ValueError(f"Unknown domain: {domain}. Available: {list(self.templates.keys())}")

            self.active_domain = domain
            self.active_template = self.templates[domain]
        else:
            self.active_domain = None  # Will detect from transcription
            self.active_template = None

        # Initialize template with defaults
        self.current_data = self.active_template.to_dict() if self.active_template else {}
        self.transcription_buffer = ""

        print(f"[LiveScratchpad] Recording started for domain: {domain}")
        self._notify_update()

    async def update_from_transcription(self, new_transcription: str, detected_domain: Optional[str] = None):
        """
        Update template from new transcription text.

        Uses NLP to extract structured data from natural language.

        Args:
            new_transcription: New transcription text
            detected_domain: Pre-detected domain (optional)
        """
        self.transcription_buffer = new_transcription

        # Use detected domain if provided
        if detected_domain and detected_domain in self.templates:
            if self.active_domain != detected_domain:
                self.active_domain = detected_domain
                self.active_template = self.templates[detected_domain]
                self.current_data = self.active_template.to_dict()
                print(f"[LiveScratchpad] Domain set to: {detected_domain}")

        # Extract data based on domain
        if self.active_template:
            extracted = self._extract_from_transcription(
                self.transcription_buffer,
                self.active_template
            )
            # Merge extracted data with current data (don't overwrite manual edits)
            for key, value in extracted.items():
                if value and (not self.current_data.get(key) or self.current_data[key] == ""):
                    self.current_data[key] = value

        self._notify_update()

    def _extract_from_transcription(
        self,
        text: str,
        template: DomainTemplate
    ) -> Dict[str, Any]:
        """Extract structured data from transcription based on template"""

        if template.name == 'recipe':
            return self._extract_recipe(text)
        elif template.name == 'budget':
            return self._extract_budget(text)
        elif template.name == 'time_tracking':
            return self._extract_time_tracking(text)
        elif template.name == 'expense':
            return self._extract_expense(text)
        elif template.name == 'bee_inspection':
            return self._extract_bee_inspection(text)
        elif template.name == 'sop':
            return self._extract_sop(text)
        else:
            return {}

    def _extract_recipe(self, text: str) -> Dict[str, Any]:
        """Extract recipe data from natural language"""
        data = {}

        # Recipe name
        name_match = re.search(
            r'(?:recipe for|making|how to make)\s+([^.!?\n]+)',
            text, re.IGNORECASE
        )
        if name_match:
            data['recipe_name'] = name_match.group(1).strip()

        # Servings
        servings_match = re.search(r'(?:serves?|servings?:?)\s*(\d+)', text, re.IGNORECASE)
        if servings_match:
            data['servings'] = int(servings_match.group(1))

        # Time
        prep_match = re.search(r'prep(?:\s+time)?:?\s*(\d+)\s*(min|hour)', text, re.IGNORECASE)
        if prep_match:
            data['prep_time'] = f"{prep_match.group(1)} {prep_match.group(2)}"

        cook_match = re.search(r'cook(?:\s+time)?:?\s*(\d+)\s*(min|hour)', text, re.IGNORECASE)
        if cook_match:
            data['cook_time'] = f"{cook_match.group(1)} {cook_match.group(2)}"

        # Ingredients (basic extraction)
        ingredient_pattern = r'(\d+(?:\.\d+)?)\s*(cup|tablespoon|teaspoon|tbsp|tsp|ounce|oz|gram|g|pound|lb)s?\s+(?:of\s+)?(\w+(?:\s+\w+)?)'
        ingredients = re.findall(ingredient_pattern, text, re.IGNORECASE)
        if ingredients:
            data['ingredients'] = [f"{amt} {unit} {ing}" for amt, unit, ing in ingredients]

        # Instructions (split by numbered steps or action verbs)
        instruction_pattern = r'(?:step\s+\d+[:.]?|first|then|next|finally)[:\s]+([^.!?]+[.!?])'
        instructions = re.findall(instruction_pattern, text, re.IGNORECASE)
        if instructions:
            data['instructions'] = [inst.strip() for inst in instructions]

        return data

    def _extract_budget(self, text: str) -> Dict[str, Any]:
        """Extract budget entry from natural language"""
        data = {}

        # Amount (look for currency symbols or numbers followed by 'dollars')
        amount_patterns = [
            r'\$(\d+(?:\.\d{2})?)',  # $45.99
            r'(\d+(?:\.\d{2})?)\s*dollars?',  # 45 dollars
            r'(\d+(?:\.\d{2})?)\s*bucks',  # 45 bucks
        ]
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['amount'] = match.group(1)
                break

        # Category keywords
        category_keywords = {
            'groceries': r'\b(groceries|grocery|food shopping|supermarket)\b',
            'dining': r'\b(restaurant|dining|eating out|lunch out|dinner out)\b',
            'transport': r'\b(gas|fuel|uber|lyft|taxi|parking|transit)\b',
            'utilities': r'\b(electric|water|gas bill|internet|phone bill)\b',
            'entertainment': r'\b(movie|concert|tickets|streaming|netflix)\b',
        }
        for category, pattern in category_keywords.items():
            if re.search(pattern, text, re.IGNORECASE):
                data['category'] = category
                break

        # Description (everything before amount or after 'for')
        desc_match = re.search(r'(?:for|bought|spent on)\s+([^$\d]+?)(?:\s+\$|\s+\d+|$)', text, re.IGNORECASE)
        if desc_match:
            data['description'] = desc_match.group(1).strip()

        return data

    def _extract_time_tracking(self, text: str) -> Dict[str, Any]:
        """Extract time tracking from natural language"""
        data = {}

        # Project name
        project_match = re.search(r'(?:worked on|project:?)\s+([^,.\n]+)', text, re.IGNORECASE)
        if project_match:
            data['project'] = project_match.group(1).strip()

        # Task
        task_match = re.search(r'(?:task:?|doing)\s+([^,.\n]+)', text, re.IGNORECASE)
        if task_match:
            data['task'] = task_match.group(1).strip()

        # Duration
        duration_patterns = [
            r'(\d+(?:\.\d+)?)\s*(hour|hr)s?',
            r'(\d+)\s*(min|minute)s?',
        ]
        for pattern in duration_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['duration'] = f"{match.group(1)} {match.group(2)}"
                break

        # Time range
        time_range = re.search(r'(\d{1,2}:\d{2})\s*(?:to|-)\s*(\d{1,2}:\d{2})', text)
        if time_range:
            data['start_time'] = time_range.group(1)
            data['end_time'] = time_range.group(2)

        return data

    def _extract_expense(self, text: str) -> Dict[str, Any]:
        """Extract expense from natural language"""
        # Similar to budget but with vendor and receipt details
        data = self._extract_budget(text)  # Reuse budget extraction

        # Vendor/merchant
        vendor_match = re.search(r'(?:at|from|vendor:?)\s+([^,.\n$\d]+)', text, re.IGNORECASE)
        if vendor_match:
            data['vendor'] = vendor_match.group(1).strip()

        # Receipt number
        receipt_match = re.search(r'(?:receipt|confirmation|order)(?:\s+number)?:?\s*([A-Z0-9-]+)', text, re.IGNORECASE)
        if receipt_match:
            data['receipt_number'] = receipt_match.group(1)

        return data

    def _extract_bee_inspection(self, text: str) -> Dict[str, Any]:
        """Extract bee inspection from natural language"""
        data = {}

        # Hive ID
        hive_match = re.search(r'(?:hive|colony)\s+([A-Z0-9]+|\w+)', text, re.IGNORECASE)
        if hive_match:
            data['hive_id'] = hive_match.group(1)

        # Temperature
        temp_match = re.search(r'(-?\d+)\s*(?:degrees?|°|F|C)', text, re.IGNORECASE)
        if temp_match:
            data['temperature'] = float(temp_match.group(1))

        # Queen seen
        if re.search(r'\b(?:saw|spotted|found).*queen\b', text, re.IGNORECASE):
            data['queen_seen'] = 'yes'
        elif re.search(r'\bno.*queen\b|\bqueen.*not\b', text, re.IGNORECASE):
            data['queen_seen'] = 'no'

        # Brood pattern
        brood_keywords = {
            'excellent': r'\bexcellent.*brood\b',
            'good': r'\bgood.*brood\b|\bhealthy.*brood\b',
            'spotty': r'\bspotty.*brood\b|\bpatchy.*brood\b',
            'poor': r'\bpoor.*brood\b|\bweak.*brood\b',
        }
        for rating, pattern in brood_keywords.items():
            if re.search(pattern, text, re.IGNORECASE):
                data['brood_pattern'] = rating
                break

        # Pests
        pests = []
        if re.search(r'\bvarroa\b|\bmites?\b', text, re.IGNORECASE):
            pests.append('varroa')
        if re.search(r'\bhive beetle\b', text, re.IGNORECASE):
            pests.append('hive beetle')
        if pests:
            data['pests_observed'] = pests

        # Actions needed
        actions = []
        if re.search(r'\bfeed\b|\bsugar\b|\bsyrup\b', text, re.IGNORECASE):
            actions.append('feed')
        if re.search(r'\btreat\b|\bmedication\b', text, re.IGNORECASE):
            actions.append('treat')
        if actions:
            data['actions_needed'] = actions

        return data

    def _extract_sop(self, text: str) -> Dict[str, Any]:
        """Extract SOP from natural language"""
        data = {}

        # Title
        title_match = re.search(r'(?:SOP|procedure|protocol)\s+for\s+([^.!?\n]+)', text, re.IGNORECASE)
        if title_match:
            data['title'] = title_match.group(1).strip()

        # Purpose
        purpose_match = re.search(r'(?:purpose|goal|objective):?\s*([^.!?\n]+)', text, re.IGNORECASE)
        if purpose_match:
            data['purpose'] = purpose_match.group(1).strip()

        # Steps (numbered or action verbs)
        step_patterns = [
            r'(?:step|stage)\s+\d+[:.]?\s*([^.!?\n]+)',
            r'(?:first|then|next|after that|finally)[:.]?\s*([^.!?\n]+)',
        ]
        steps = []
        for pattern in step_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            steps.extend([m.strip() for m in matches])

        if steps:
            data['procedure_steps'] = steps

        # Responsibilities
        if re.search(r'\b(manager|supervisor|lead|coordinator)\b', text, re.IGNORECASE):
            data['responsibilities'] = ['designated staff member']

        # Safety warnings
        safety_keywords = ['warning', 'caution', 'danger', 'hazard', 'safety', 'protect']
        if any(re.search(rf'\b{kw}\b', text, re.IGNORECASE) for kw in safety_keywords):
            data['safety_warnings'] = ['See safety notes in procedure']

        # Frequency
        freq_patterns = {
            'daily': r'\b(daily|every day|each day)\b',
            'weekly': r'\b(weekly|every week|once a week)\b',
            'monthly': r'\b(monthly|every month|once a month)\b',
            'seasonal': r'\b(seasonal|spring|summer|fall|winter)\b',
        }
        for freq, pattern in freq_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                data['frequency'] = freq
                break

        return data

    def update_field(self, field_name: str, value: Any):
        """Manually update field (from UI)"""
        self.current_data[field_name] = value
        self._notify_update()

    def get_current_state(self) -> Dict[str, Any]:
        """Get current template state (for UI rendering)"""
        validation_errors = []
        if self.active_template:
            validation_errors = self.active_template.validate(self.current_data)

        return {
            'domain': self.active_domain,
            'template': self.active_template.name if self.active_template else None,
            'template_fields': [
                {
                    'name': f.name,
                    'type': f.field_type,
                    'required': f.required,
                    'suggestions': f.suggestions
                }
                for f in self.active_template.fields
            ] if self.active_template else [],
            'data': self.current_data,
            'transcription': self.transcription_buffer,
            'validation_errors': validation_errors,
            'is_valid': len(validation_errors) == 0
        }

    async def finalize(self) -> MemoryShard:
        """
        Finalize scratchpad and create memory shard.

        Returns validated, structured shard ready for ingestion.
        """
        if not self.active_template:
            raise ValueError("No active template")

        # Validate
        errors = self.active_template.validate(self.current_data)
        if errors:
            raise ValueError(f"Validation failed: {errors}")

        # Create shard
        shard = MemoryShard(
            id=f"{self.active_domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            text=self.transcription_buffer,
            episode=self.active_domain,
            entities=[],  # Could extract from current_data
            motifs=[self.active_domain],
            metadata={
                'domain': self.active_domain,
                'structured_data': self.current_data,
                'finalized_at': datetime.now().isoformat()
            }
        )

        print(f"[LiveScratchpad] Finalized {self.active_domain} entry")
        return shard

    def clear(self):
        """Clear current session"""
        self.active_domain = None
        self.active_template = None
        self.current_data = {}
        self.transcription_buffer = ""
        self._notify_update()
