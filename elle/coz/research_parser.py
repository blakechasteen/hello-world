"""
Research Notes Parser for Elle Core

Reads coz/research_notes.md and extracts:
- Experiment results and trials
- Formulas and ratios
- Batch results
- Insights for knowledge base ingestion

Created: 2025-11-15
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class ExperimentResult:
    """Result from an experiment trial"""
    batch_number: str
    description: str
    result: str
    notes: str = ""
    status: str = "completed"  # completed, in_progress, pending
    value: Optional[str] = None  # Extracted numerical value if applicable

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "batch_number": self.batch_number,
            "description": self.description,
            "result": self.result,
            "notes": self.notes,
            "status": self.status,
            "value": self.value,
        }


@dataclass
class ResearchNote:
    """Research note with experiments"""
    title: str
    category: str
    experiments: List[ExperimentResult]
    formulas: List[Dict] = None
    recommendations: List[str] = None

    def __post_init__(self):
        if self.formulas is None:
            self.formulas = []
        if self.recommendations is None:
            self.recommendations = []

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "category": self.category,
            "experiment_count": len(self.experiments),
            "experiments": [e.to_dict() for e in self.experiments],
            "formulas": self.formulas,
            "recommendations": self.recommendations,
        }


class ResearchParser:
    """Parser for research_notes.md file"""

    def __init__(self, research_path: str = "coz/research_notes.md"):
        self.research_path = Path(research_path)
        self.research_notes: List[ResearchNote] = []
        self.categories: Dict[str, List[ResearchNote]] = {}

    def parse(self) -> List[ResearchNote]:
        """Parse research_notes.md file"""
        if not self.research_path.exists():
            raise FileNotFoundError(f"Research notes file not found: {self.research_path}")

        with open(self.research_path, 'r') as f:
            content = f.read()

        self.research_notes = self._parse_research_sections(content)

        # Organize by category
        for note in self.research_notes:
            if note.category not in self.categories:
                self.categories[note.category] = []
            self.categories[note.category].append(note)

        return self.research_notes

    def _parse_research_sections(self, content: str) -> List[ResearchNote]:
        """Parse research sections from markdown"""
        notes = []

        # Find all level-2 headers (## Topic)
        sections = re.split(r'^## ', content, flags=re.MULTILINE)[1:]

        for section in sections:
            lines = section.split('\n')
            if not lines:
                continue

            # First line is the title
            title = lines[0].strip()

            # Extract category (usually from first part of title)
            category = self._extract_category(title)

            # Parse experiments
            experiments = self._parse_experiments(section)

            # Parse formulas if any
            formulas = self._parse_formulas(section)

            # Extract recommendations
            recommendations = self._extract_recommendations(section)

            note = ResearchNote(
                title=title,
                category=category,
                experiments=experiments,
                formulas=formulas,
                recommendations=recommendations,
            )
            notes.append(note)

        return notes

    def _extract_category(self, title: str) -> str:
        """Extract category from title"""
        # Remove common suffixes
        title_clean = title.replace("Flavor Trials", "").replace("Trials", "").replace("Mix", "").strip()
        return title_clean

    def _parse_experiments(self, section: str) -> List[ExperimentResult]:
        """Parse individual experiment trials/batches"""
        experiments = []

        # Look for batch patterns like "Batch 01:", "- Batch XX:", etc.
        batch_pattern = r'- (Batch \d+|Batch [A-Za-z0-9]+|[Bb]atch \d+):\s*(.+?)(?=\n- |$)'
        matches = re.finditer(batch_pattern, section, re.DOTALL)

        for match in matches:
            batch_id = match.group(1).strip()
            content = match.group(2).strip()

            # Extract description and result
            # Usually format is: "Description — Result."
            if '—' in content:
                parts = content.split('—', 1)
                description = parts[0].strip()
                result = parts[1].strip()
            elif ' - ' in content:
                parts = content.split(' - ', 1)
                description = parts[0].strip()
                result = parts[1].strip()
            else:
                description = content
                result = ""

            # Try to extract numerical values
            value = self._extract_numerical_value(result)

            experiment = ExperimentResult(
                batch_number=batch_id,
                description=description,
                result=result,
                value=value,
            )
            experiments.append(experiment)

        return experiments

    def _extract_numerical_value(self, text: str) -> Optional[str]:
        """Extract numerical values from result text"""
        # Look for patterns like "3 oz", "10 days", "3:1 ratio", etc.
        match = re.search(r'(\d+(?:\.\d+)?(?:\s*(?:oz|lb|ml|l|days?|hours?|minutes?|ratio|%|parts?))?)', text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _parse_formulas(self, section: str) -> List[Dict]:
        """Extract formulas and ratios"""
        formulas = []

        # Look for ratio patterns like "3 parts : 1 part" or "3:1"
        ratio_pattern = r'(\d+(?:\.\d+)?)\s*(?:parts?|:)\s*(\d+(?:\.\d+)?)'
        matches = re.finditer(ratio_pattern, section)

        for match in matches:
            formula = {
                "type": "ratio",
                "left": match.group(1),
                "right": match.group(2),
                "ratio": f"{match.group(1)}:{match.group(2)}",
            }
            formulas.append(formula)

        # Look for yield patterns
        yield_pattern = r'yields?:\s*(\d+(?:\.\d+)?)\s*(\w+)'
        matches = re.finditer(yield_pattern, section, re.IGNORECASE)

        for match in matches:
            formula = {
                "type": "yield",
                "amount": match.group(1),
                "unit": match.group(2),
            }
            formulas.append(formula)

        return formulas

    def _extract_recommendations(self, section: str) -> List[str]:
        """Extract recommendations from section"""
        recommendations = []

        # Look for improvement/recommendation patterns
        improvement_patterns = [
            r'[Ii]mproved?\s+(.+?)(?:\.|$)',
            r'[Ss]hould\s+(.+?)(?:\.|$)',
            r'[Rr]ecommend\s+(.+?)(?:\.|$)',
        ]

        for pattern in improvement_patterns:
            matches = re.finditer(pattern, section)
            for match in matches:
                rec = match.group(1).strip()
                if rec and rec not in recommendations:
                    recommendations.append(rec)

        return recommendations

    def get_experiments_by_category(self, category: str) -> List[ExperimentResult]:
        """Get all experiments in a category"""
        experiments = []
        for note in self.categories.get(category, []):
            experiments.extend(note.experiments)
        return experiments

    def get_category_summary(self, category: str) -> Dict:
        """Get summary for a category"""
        notes = self.categories.get(category, [])
        experiments = self.get_experiments_by_category(category)

        return {
            "category": category,
            "research_projects": len(notes),
            "total_experiments": len(experiments),
            "projects": [n.title for n in notes],
            "experiments": [e.to_dict() for e in experiments],
        }

    def get_all_formulas(self) -> List[Dict]:
        """Get all formulas and ratios"""
        formulas = []
        for note in self.research_notes:
            formulas.extend(note.formulas)
        return formulas

    def get_all_recommendations(self) -> List[str]:
        """Get all recommendations"""
        recommendations = []
        for note in self.research_notes:
            recommendations.extend(note.recommendations)
        return recommendations

    def search_experiments(self, keyword: str) -> List[ExperimentResult]:
        """Search experiments by keyword"""
        results = []
        keyword_lower = keyword.lower()

        for note in self.research_notes:
            for exp in note.experiments:
                if (keyword_lower in exp.description.lower() or
                    keyword_lower in exp.result.lower() or
                    keyword_lower in exp.notes.lower()):
                    results.append(exp)

        return results

    def get_summary(self) -> Dict:
        """Get research summary"""
        total_experiments = sum(len(note.experiments) for note in self.research_notes)
        total_formulas = sum(len(note.formulas) for note in self.research_notes)
        total_recommendations = sum(len(note.recommendations) for note in self.research_notes)

        return {
            "research_projects": len(self.research_notes),
            "total_experiments": total_experiments,
            "total_formulas": total_formulas,
            "total_recommendations": total_recommendations,
            "categories": list(self.categories.keys()),
            "by_category": {
                cat: len(notes) for cat, notes in self.categories.items()
            },
            "projects": [n.to_dict() for n in self.research_notes],
        }

    def to_knowledge_base_entries(self) -> List[Dict]:
        """Convert research notes to knowledge base entries for HoloLoom ingestion"""
        entries = []

        for note in self.research_notes:
            entry = {
                "type": "research_note",
                "title": note.title,
                "category": note.category,
                "content": f"Research: {note.title}\n\n",
                "experiments": [],
                "formulas": note.formulas,
                "recommendations": note.recommendations,
            }

            # Add experiment details
            for exp in note.experiments:
                exp_text = f"\n{exp.batch_number}: {exp.description} — {exp.result}"
                entry["content"] += exp_text
                entry["experiments"].append(exp.to_dict())

            entries.append(entry)

        return entries
