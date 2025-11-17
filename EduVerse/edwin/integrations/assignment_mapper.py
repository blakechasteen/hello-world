"""
Assignment Mapper

Maps LMS assignments to EdWIN learning objectives.

Author: Agent D
Date: 2025-11-15
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import os


class GradingScheme(Enum):
    """Grading scheme for converting EdWIN mastery to LMS grades"""
    MASTERY_PERCENTAGE = "mastery_percentage"  # 0.85 → 85%
    POINTS = "points"  # 0.85 × max_points
    PASS_FAIL = "pass_fail"  # >=0.7 → complete
    LETTER_GRADE = "letter_grade"  # 0.85 → B


@dataclass
class ObjectiveMapping:
    """Mapping between LMS assignment and EdWIN objectives"""
    lms_assignment_id: str
    lms_assignment_name: str
    edwin_objectives: List[str]  # List of objective IDs
    grading_scheme: GradingScheme
    grading_weights: Dict[str, float]  # objective_id → weight
    max_points: float = 100.0
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class AssignmentMapper:
    """
    Maps LMS assignments to EdWIN learning objectives.

    Features:
    - Manual mapping interface
    - Automatic mapping suggestions (using LLM)
    - Multiple objectives per assignment
    - Custom grading weights
    - Due date synchronization
    """

    # Grading scheme implementations
    GRADING_SCHEMES: Dict[GradingScheme, Callable] = {
        GradingScheme.MASTERY_PERCENTAGE: lambda score, max_points=100: score * 100,
        GradingScheme.POINTS: lambda score, max_points=100: score * max_points,
        GradingScheme.PASS_FAIL: lambda score, max_points=100: "complete" if score >= 0.7 else "incomplete",
        GradingScheme.LETTER_GRADE: lambda score, max_points=100: AssignmentMapper._convert_to_letter(score),
    }

    def __init__(self, mappings_file: str = "./assignment_mappings.json"):
        """
        Initialize assignment mapper.

        Args:
            mappings_file: Path to store assignment mappings
        """
        self.mappings_file = mappings_file
        self.mappings: Dict[str, ObjectiveMapping] = {}
        self._load_mappings()

    def _load_mappings(self):
        """Load mappings from file"""
        if not os.path.exists(self.mappings_file):
            return

        try:
            with open(self.mappings_file, "r") as f:
                data = json.load(f)

            for assignment_id, mapping_data in data.items():
                # Convert string back to enum
                grading_scheme = GradingScheme(mapping_data["grading_scheme"])

                self.mappings[assignment_id] = ObjectiveMapping(
                    lms_assignment_id=mapping_data["lms_assignment_id"],
                    lms_assignment_name=mapping_data["lms_assignment_name"],
                    edwin_objectives=mapping_data["edwin_objectives"],
                    grading_scheme=grading_scheme,
                    grading_weights=mapping_data["grading_weights"],
                    max_points=mapping_data.get("max_points", 100.0),
                    created_at=datetime.fromisoformat(mapping_data["created_at"]),
                )
        except Exception as e:
            print(f"Error loading mappings: {e}")

    def _save_mappings(self):
        """Save mappings to file"""
        data = {}

        for assignment_id, mapping in self.mappings.items():
            data[assignment_id] = {
                "lms_assignment_id": mapping.lms_assignment_id,
                "lms_assignment_name": mapping.lms_assignment_name,
                "edwin_objectives": mapping.edwin_objectives,
                "grading_scheme": mapping.grading_scheme.value,
                "grading_weights": mapping.grading_weights,
                "max_points": mapping.max_points,
                "created_at": mapping.created_at.isoformat(),
            }

        os.makedirs(os.path.dirname(self.mappings_file) if os.path.dirname(self.mappings_file) else ".", exist_ok=True)
        with open(self.mappings_file, "w") as f:
            json.dump(data, f, indent=2)

    def create_mapping(
        self,
        lms_assignment_id: str,
        lms_assignment_name: str,
        edwin_objectives: List[str],
        grading_scheme: GradingScheme = GradingScheme.MASTERY_PERCENTAGE,
        grading_weights: Optional[Dict[str, float]] = None,
        max_points: float = 100.0,
    ) -> ObjectiveMapping:
        """
        Create mapping between LMS assignment and EdWIN objectives.

        Args:
            lms_assignment_id: LMS assignment ID
            lms_assignment_name: LMS assignment name
            edwin_objectives: List of EdWIN objective IDs
            grading_scheme: Grading scheme to use
            grading_weights: Optional custom weights (default: equal weight)
            max_points: Maximum points for assignment

        Returns:
            ObjectiveMapping
        """
        # Default to equal weights
        if grading_weights is None:
            equal_weight = 1.0 / len(edwin_objectives)
            grading_weights = {obj_id: equal_weight for obj_id in edwin_objectives}

        # Validate weights sum to 1.0
        total_weight = sum(grading_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Grading weights must sum to 1.0, got {total_weight}")

        mapping = ObjectiveMapping(
            lms_assignment_id=lms_assignment_id,
            lms_assignment_name=lms_assignment_name,
            edwin_objectives=edwin_objectives,
            grading_scheme=grading_scheme,
            grading_weights=grading_weights,
            max_points=max_points,
        )

        self.mappings[lms_assignment_id] = mapping
        self._save_mappings()

        return mapping

    def get_mapping(self, lms_assignment_id: str) -> Optional[ObjectiveMapping]:
        """Get mapping for LMS assignment"""
        return self.mappings.get(lms_assignment_id)

    def delete_mapping(self, lms_assignment_id: str):
        """Delete mapping"""
        if lms_assignment_id in self.mappings:
            del self.mappings[lms_assignment_id]
            self._save_mappings()

    def calculate_grade(
        self,
        lms_assignment_id: str,
        objective_scores: Dict[str, float],
    ) -> float:
        """
        Calculate LMS grade from EdWIN objective scores.

        Args:
            lms_assignment_id: LMS assignment ID
            objective_scores: Dict of objective_id → mastery score (0.0-1.0)

        Returns:
            LMS grade (format depends on grading scheme)
        """
        mapping = self.get_mapping(lms_assignment_id)
        if not mapping:
            raise ValueError(f"No mapping found for assignment {lms_assignment_id}")

        # Calculate weighted average
        total_score = 0.0
        for obj_id in mapping.edwin_objectives:
            score = objective_scores.get(obj_id, 0.0)
            weight = mapping.grading_weights.get(obj_id, 0.0)
            total_score += score * weight

        # Apply grading scheme
        grading_func = self.GRADING_SCHEMES[mapping.grading_scheme]
        return grading_func(total_score, mapping.max_points)

    @staticmethod
    def _convert_to_letter(score: float) -> str:
        """Convert mastery score to letter grade"""
        if score >= 0.93:
            return "A"
        elif score >= 0.90:
            return "A-"
        elif score >= 0.87:
            return "B+"
        elif score >= 0.83:
            return "B"
        elif score >= 0.80:
            return "B-"
        elif score >= 0.77:
            return "C+"
        elif score >= 0.73:
            return "C"
        elif score >= 0.70:
            return "C-"
        elif score >= 0.67:
            return "D+"
        elif score >= 0.63:
            return "D"
        elif score >= 0.60:
            return "D-"
        else:
            return "F"

    def suggest_mapping(
        self,
        assignment_name: str,
        assignment_description: str,
        available_objectives: List[Dict[str, str]],
    ) -> List[str]:
        """
        Suggest EdWIN objectives for an LMS assignment.

        Uses simple keyword matching. Could be enhanced with LLM.

        Args:
            assignment_name: LMS assignment name
            assignment_description: LMS assignment description
            available_objectives: List of dicts with 'id', 'name', 'description'

        Returns:
            List of suggested objective IDs
        """
        # Simple keyword matching (could be enhanced with LLM)
        assignment_text = f"{assignment_name} {assignment_description}".lower()

        suggestions = []
        for obj in available_objectives:
            obj_text = f"{obj['name']} {obj['description']}".lower()

            # Count keyword matches
            keywords = set(obj_text.split())
            matches = sum(1 for word in assignment_text.split() if word in keywords)

            if matches > 2:  # Threshold for suggestion
                suggestions.append(obj["id"])

        return suggestions[:5]  # Top 5 suggestions

    def get_all_mappings(self) -> List[ObjectiveMapping]:
        """Get all mappings"""
        return list(self.mappings.values())

    def get_mappings_for_objectives(self, objective_ids: List[str]) -> List[ObjectiveMapping]:
        """Get all mappings that include any of the given objectives"""
        results = []

        for mapping in self.mappings.values():
            if any(obj_id in mapping.edwin_objectives for obj_id in objective_ids):
                results.append(mapping)

        return results

    def export_mappings(self, export_path: str):
        """Export mappings to file"""
        with open(export_path, "w") as f:
            data = {
                assignment_id: {
                    "assignment_name": mapping.lms_assignment_name,
                    "objectives": mapping.edwin_objectives,
                    "grading_scheme": mapping.grading_scheme.value,
                    "weights": mapping.grading_weights,
                }
                for assignment_id, mapping in self.mappings.items()
            }
            json.dump(data, f, indent=2)

    def import_mappings(self, import_path: str):
        """Import mappings from file"""
        with open(import_path, "r") as f:
            data = json.load(f)

        for assignment_id, mapping_data in data.items():
            self.create_mapping(
                lms_assignment_id=assignment_id,
                lms_assignment_name=mapping_data["assignment_name"],
                edwin_objectives=mapping_data["objectives"],
                grading_scheme=GradingScheme(mapping_data["grading_scheme"]),
                grading_weights=mapping_data["weights"],
            )
