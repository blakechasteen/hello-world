"""
Mediation Department

Handles:
- Conflict resolution between residents
- Relationship repair strategies
- Fair blame attribution
- Intervention recommendations

Uses causal reasoning to:
- Understand root causes of conflicts
- Suggest effective interventions
- Predict relationship trajectories
- Recommend communication strategies
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from HoloLoom.departments.base import BaseDepartment
from HoloLoom.departments.protocol import DepartmentRequest, DepartmentResponse

from ..causal.blame import BlameAttributionSystem, BlameReport, quick_blame_attribution
from ..causal.interventions import NeuroHoodInterventionEngine
from ..causal.relationship_scm import RelationshipSCM, extract_relationship_data, simulate_relationship_outcome

logger = logging.getLogger(__name__)


@dataclass
class MediationCase:
    """Mediation case between two residents."""
    case_id: str
    resident_a: str
    resident_b: str
    description: str
    status: str = "open"  # "open", "in_mediation", "resolved", "failed"
    timestamp: float = 0.0
    blame_report: Optional[BlameReport] = None
    recommended_interventions: List[str] = None


class MediationDepartment(BaseDepartment):
    """
    Mediation Department for conflict resolution.

    Uses causal reasoning to:
    - Fairly attribute blame in conflicts
    - Suggest effective interventions
    - Predict relationship trajectories
    - Monitor resolution progress
    """

    def __init__(self, scm: Optional[RelationshipSCM] = None):
        """
        Initialize Mediation Department.

        Args:
            scm: Optional trained RelationshipSCM for causal reasoning
        """
        super().__init__(name="Mediation", version="1.0.0")
        self.scm = scm
        self.blame_system = BlameAttributionSystem(scm) if scm else None
        self.intervention_engine = NeuroHoodInterventionEngine(scm) if scm else None

        # Mediation cases
        self.cases: List[MediationCase] = []

    async def process(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Process mediation request.

        Supported request types:
        - "open_case": Open a new mediation case
        - "analyze_conflict": Analyze conflict with blame attribution
        - "suggest_interventions": Suggest conflict resolution strategies
        - "predict_trajectory": Predict relationship future
        - "close_case": Close a mediation case
        - "get_statistics": Get department statistics
        """
        request_type = request.request_type

        if request_type == "open_case":
            return await self._open_case(request)
        elif request_type == "analyze_conflict":
            return await self._analyze_conflict(request)
        elif request_type == "suggest_interventions":
            return await self._suggest_interventions(request)
        elif request_type == "predict_trajectory":
            return await self._predict_trajectory(request)
        elif request_type == "close_case":
            return await self._close_case(request)
        elif request_type == "get_statistics":
            return await self._get_statistics(request)
        else:
            return DepartmentResponse(
                success=False,
                error=f"Unknown request type: {request_type}"
            )

    async def _open_case(self, request: DepartmentRequest) -> DepartmentResponse:
        """Open a new mediation case."""
        data = request.data

        resident_a = data.get("resident_a")
        resident_b = data.get("resident_b")
        description = data.get("description")

        if not all([resident_a, resident_b, description]):
            return DepartmentResponse(
                success=False,
                error="Missing required fields"
            )

        case = MediationCase(
            case_id=f"M-{len(self.cases)+1:04d}",
            resident_a=resident_a,
            resident_b=resident_b,
            description=description,
            timestamp=data.get("timestamp", 0.0)
        )

        self.cases.append(case)

        logger.info(f"Opened mediation case: {case.case_id}")

        return DepartmentResponse(
            success=True,
            data={
                "case_id": case.case_id,
                "message": f"Mediation case opened: {case.case_id}"
            }
        )

    async def _analyze_conflict(self, request: DepartmentRequest) -> DepartmentResponse:
        """Analyze conflict with blame attribution."""
        data = request.data
        case_id = data.get("case_id")
        state = data.get("state")  # NeuroHoodState

        if not case_id:
            return DepartmentResponse(
                success=False,
                error="Missing case_id"
            )

        case = next((c for c in self.cases if c.case_id == case_id), None)
        if not case:
            return DepartmentResponse(
                success=False,
                error=f"Case not found: {case_id}"
            )

        if not self.scm or not state:
            return DepartmentResponse(
                success=False,
                error="Causal reasoning requires SCM and state data"
            )

        # Find relationship
        relationship_key = self._get_relationship_key(
            case.resident_a,
            case.resident_b,
            state
        )

        if not relationship_key:
            return DepartmentResponse(
                success=False,
                error=f"No relationship found between {case.resident_a} and {case.resident_b}"
            )

        relationship = state.relationships[relationship_key]
        factual_data = extract_relationship_data(
            relationship,
            state.residents,
            timestamp=case.timestamp
        )

        # Compute blame attribution
        blame_report = quick_blame_attribution(
            self.scm,
            factual_data,
            resident_a_name=case.resident_a,
            resident_b_name=case.resident_b,
            conflict_description=case.description
        )

        case.blame_report = blame_report

        logger.info(f"Analyzed conflict for {case_id}")

        return DepartmentResponse(
            success=True,
            data={
                "case_id": case_id,
                "blame_report": blame_report.format(),
                "primary_cause": blame_report.primary_cause,
                "total_blame_explained": blame_report.total_blame_explained
            }
        )

    async def _suggest_interventions(self, request: DepartmentRequest) -> DepartmentResponse:
        """Suggest conflict resolution interventions."""
        data = request.data
        case_id = data.get("case_id")
        state = data.get("state")

        if not case_id:
            return DepartmentResponse(
                success=False,
                error="Missing case_id"
            )

        case = next((c for c in self.cases if c.case_id == case_id), None)
        if not case:
            return DepartmentResponse(
                success=False,
                error=f"Case not found: {case_id}"
            )

        if not self.intervention_engine or not state:
            return DepartmentResponse(
                success=False,
                error="Intervention reasoning requires SCM and state data"
            )

        # Find relationship
        relationship_key = self._get_relationship_key(
            case.resident_a,
            case.resident_b,
            state
        )

        if not relationship_key:
            return DepartmentResponse(
                success=False,
                error=f"No relationship found"
            )

        relationship = state.relationships[relationship_key]
        factual_data = extract_relationship_data(
            relationship,
            state.residents,
            timestamp=case.timestamp
        )

        # Test various interventions
        interventions = [
            {"communication_quality": 0.9},  # Improve communication
            {"conflict_intensity": 0.1},     # Reduce conflict
            {"communication_quality": 0.9, "conflict_intensity": 0.1},  # Both
        ]

        context = {
            "personality_compatibility": factual_data.personality_compatibility
        }

        ranked_interventions = self.intervention_engine.compare_interventions(
            interventions,
            context=context,
            outcome_variable="relationship_strength"
        )

        # Format recommendations
        recommendations = []
        for i, (intervention, outcome) in enumerate(ranked_interventions, 1):
            desc = []
            if "communication_quality" in intervention:
                desc.append("Improve communication (apology, respectful dialogue)")
            if "conflict_intensity" in intervention:
                desc.append("Reduce conflict triggers (noise reduction, boundaries)")

            recommendations.append({
                "rank": i,
                "intervention": ", ".join(desc),
                "predicted_relationship_strength": f"{outcome:.2f}",
                "details": intervention
            })

        case.recommended_interventions = recommendations

        logger.info(f"Generated {len(recommendations)} intervention recommendations for {case_id}")

        return DepartmentResponse(
            success=True,
            data={
                "case_id": case_id,
                "recommendations": recommendations
            }
        )

    async def _predict_trajectory(self, request: DepartmentRequest) -> DepartmentResponse:
        """Predict relationship trajectory."""
        data = request.data
        case_id = data.get("case_id")
        state = data.get("state")
        n_steps = data.get("n_steps", 30)  # 30 days default

        if not case_id:
            return DepartmentResponse(
                success=False,
                error="Missing case_id"
            )

        case = next((c for c in self.cases if c.case_id == case_id), None)
        if not case:
            return DepartmentResponse(
                success=False,
                error=f"Case not found: {case_id}"
            )

        if not self.scm or not state:
            return DepartmentResponse(
                success=False,
                error="Trajectory prediction requires SCM and state data"
            )

        # Find relationship
        relationship_key = self._get_relationship_key(
            case.resident_a,
            case.resident_b,
            state
        )

        if not relationship_key:
            return DepartmentResponse(
                success=False,
                error=f"No relationship found"
            )

        relationship = state.relationships[relationship_key]
        factual_data = extract_relationship_data(
            relationship,
            state.residents,
            timestamp=case.timestamp
        )

        # Simulate trajectory
        trajectory = simulate_relationship_outcome(
            self.scm,
            personality_compatibility=factual_data.personality_compatibility,
            communication_quality=factual_data.communication_quality,
            conflict_intensity=factual_data.conflict_intensity,
            n_steps=n_steps
        )

        # Extract key metrics
        strengths = [t["relationship_strength"] for t in trajectory]
        conflicts = [t["conflict_intensity"] for t in trajectory]

        final_strength = strengths[-1]
        avg_conflict = sum(conflicts) / len(conflicts)

        # Determine prognosis
        if final_strength > 0.7:
            prognosis = "positive"
            message = "Relationship likely to improve over time"
        elif final_strength < 0.3:
            prognosis = "negative"
            message = "Relationship at risk of deterioration"
        else:
            prognosis = "neutral"
            message = "Relationship stable but may need support"

        logger.info(f"Predicted trajectory for {case_id}: {prognosis}")

        return DepartmentResponse(
            success=True,
            data={
                "case_id": case_id,
                "prognosis": prognosis,
                "message": message,
                "final_strength": final_strength,
                "avg_conflict": avg_conflict,
                "trajectory": trajectory
            }
        )

    async def _close_case(self, request: DepartmentRequest) -> DepartmentResponse:
        """Close a mediation case."""
        case_id = request.data.get("case_id")
        resolution = request.data.get("resolution", "resolved")  # "resolved" or "failed"

        case = next((c for c in self.cases if c.case_id == case_id), None)
        if not case:
            return DepartmentResponse(
                success=False,
                error=f"Case not found: {case_id}"
            )

        case.status = resolution

        logger.info(f"Closed case {case_id}: {resolution}")

        return DepartmentResponse(
            success=True,
            data={
                "case_id": case_id,
                "status": resolution
            }
        )

    async def _get_statistics(self, request: DepartmentRequest) -> DepartmentResponse:
        """Get department statistics."""
        total = len(self.cases)
        by_status = {}

        for c in self.cases:
            by_status[c.status] = by_status.get(c.status, 0) + 1

        success_rate = (
            by_status.get("resolved", 0) / total
            if total > 0 else 0.0
        )

        return DepartmentResponse(
            success=True,
            data={
                "total_cases": total,
                "by_status": by_status,
                "success_rate": success_rate
            }
        )

    def _get_relationship_key(
        self,
        resident_a: str,
        resident_b: str,
        state: Any
    ) -> Optional[str]:
        """Get relationship key between two residents."""
        for key, rel in state.relationships.items():
            if (rel.resident_a == resident_a and rel.resident_b == resident_b) or \
               (rel.resident_a == resident_b and rel.resident_b == resident_a):
                return key
        return None
