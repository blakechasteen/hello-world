"""
Elegant HoloLoom integration adapter for Keep.

Provides seamless bidirectional integration:
- Apiary state → HoloLoom memory shards
- HoloLoom reasoning → Beekeeping recommendations
- Natural language queries → Structured insights
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import asdict

from apps.keep.apiary import Apiary
from apps.keep.models import Hive, Colony, Inspection, Alert
from apps.keep.types import HealthStatus, QueenStatus


class ApiaryMemoryAdapter:
    """
    Adapts apiary state into HoloLoom memory shards.

    Converts beekeeping domain objects into rich memory representations
    suitable for HoloLoom's multi-scale embedding and reasoning.
    """

    def __init__(self, apiary: Apiary):
        """
        Initialize adapter.

        Args:
            apiary: Apiary to adapt
        """
        self.apiary = apiary

    def to_memory_shards(self) -> List[Any]:
        """
        Convert entire apiary state to memory shards.

        Returns:
            List of MemoryShard objects for HoloLoom ingestion
        """
        try:
            from HoloLoom.documentation.types import MemoryShard
        except ImportError:
            # Graceful degradation if HoloLoom unavailable
            return []

        shards = []

        # Apiary overview shard
        shards.append(self._create_apiary_overview_shard())

        # Hive and colony shards
        for hive_id, hive in self.apiary.hives.items():
            shards.append(self._create_hive_shard(hive))

            colony = self.apiary.get_colony(hive_id)
            if colony:
                shards.append(self._create_colony_shard(hive, colony))

        # Recent inspection shards (last 30 days)
        recent_inspections = [
            insp for insp in self.apiary.inspections
            if (datetime.now() - insp.timestamp).days <= 30
        ]
        for inspection in recent_inspections[-20:]:  # Last 20 inspections
            shards.append(self._create_inspection_shard(inspection))

        # Active alert shards
        active_alerts = self.apiary.get_active_alerts()
        for alert in active_alerts[:10]:  # Top 10 alerts
            shards.append(self._create_alert_shard(alert))

        return shards

    def _create_apiary_overview_shard(self) -> Any:
        """Create overview shard for apiary."""
        from HoloLoom.documentation.types import MemoryShard

        summary = self.apiary.get_apiary_summary()

        text = f"""
Apiary: {summary['name']}
Location: {summary['location']}

Fleet Status:
- Total Hives: {summary['total_hives']}
- Active Colonies: {summary['active_colonies']}
- Healthy Colonies: {summary['healthy_colonies']}

Management:
- Total Inspections: {summary['inspections_recorded']}
- Active Alerts: {summary['active_alerts']}
- Critical Alerts: {summary['critical_alerts']}

Production:
- Yearly Harvest: {summary['yearly_harvest_lbs']} lbs honey

Overall Health: {summary['healthy_colonies']}/{summary['active_colonies']} colonies healthy
        """.strip()

        return MemoryShard(
            id=f"apiary_{self.apiary.name}",
            text=text,
            metadata={
                "source": "keep_apiary",
                "type": "apiary_overview",
                "timestamp": datetime.now().isoformat(),
                **summary
            }
        )

    def _create_hive_shard(self, hive: Hive) -> Any:
        """Create memory shard for a hive."""
        from HoloLoom.documentation.types import MemoryShard

        text = f"""
Hive: {hive.name}
Type: {hive.hive_type.value}
Location: {hive.location}
Installed: {hive.installation_date.strftime('%Y-%m-%d')}

Notes: {hive.notes or 'No additional notes'}
        """.strip()

        return MemoryShard(
            id=f"hive_{hive.hive_id}",
            text=text,
            metadata={
                "source": "keep_hive",
                "type": "hive",
                "hive_id": hive.hive_id,
                "hive_name": hive.name,
                "hive_type": hive.hive_type.value,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def _create_colony_shard(self, hive: Hive, colony: Colony) -> Any:
        """Create memory shard for a colony."""
        from HoloLoom.documentation.types import MemoryShard

        age_info = (
            f"Queen age: {colony.queen_age_months} months"
            if colony.queen_age_months
            else "Queen age: unknown"
        )

        text = f"""
Colony in {hive.name}
Health: {colony.health_status.value.upper()}
Queen Status: {colony.queen_status.value}
{age_info}
Population: ~{colony.population_estimate:,} bees
Breed: {colony.breed}
Origin: {colony.origin}

Established: {colony.established_date.strftime('%Y-%m-%d')}
Notes: {colony.notes or 'No additional notes'}
        """.strip()

        return MemoryShard(
            id=f"colony_{colony.colony_id}",
            text=text,
            metadata={
                "source": "keep_colony",
                "type": "colony",
                "hive_id": colony.hive_id,
                "colony_id": colony.colony_id,
                "health_status": colony.health_status.value,
                "queen_status": colony.queen_status.value,
                "population": colony.population_estimate,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def _create_inspection_shard(self, inspection: Inspection) -> Any:
        """Create memory shard for an inspection."""
        from HoloLoom.documentation.types import MemoryShard

        findings = inspection.findings
        hive = self.apiary.hives.get(inspection.hive_id)
        hive_name = hive.name if hive else inspection.hive_id

        # Build findings summary
        findings_text = []
        if findings.get("queen_seen"):
            findings_text.append("- Queen observed")
        if findings.get("eggs_seen"):
            findings_text.append("- Eggs present")
        if findings.get("larvae_seen"):
            findings_text.append("- Larvae present")
        if findings.get("capped_brood_seen"):
            findings_text.append("- Capped brood present")

        frames = []
        if "frames_with_brood" in findings:
            frames.append(f"{findings['frames_with_brood']} brood")
        if "frames_with_honey" in findings:
            frames.append(f"{findings['frames_with_honey']} honey")
        if "frames_with_pollen" in findings:
            frames.append(f"{findings['frames_with_pollen']} pollen")
        if frames:
            findings_text.append(f"- Frames: {', '.join(frames)}")

        if findings.get("population_estimate"):
            findings_text.append(f"- Population: ~{findings['population_estimate']:,}")

        # Concerns
        concerns = []
        if findings.get("mites_observed"):
            concerns.append("varroa mites")
        if findings.get("beetles_observed"):
            concerns.append("small hive beetles")
        if findings.get("swarm_cells", 0) > 0:
            concerns.append(f"{findings['swarm_cells']} swarm cells")
        if findings.get("disease_signs"):
            concerns.extend(findings["disease_signs"])

        if concerns:
            findings_text.append(f"- CONCERNS: {', '.join(concerns)}")

        text = f"""
Inspection: {hive_name}
Date: {inspection.timestamp.strftime('%Y-%m-%d %H:%M')}
Type: {inspection.inspection_type.value}
Inspector: {inspection.inspector or 'Unknown'}
Weather: {inspection.weather or 'Not recorded'}
Duration: {inspection.duration_minutes or '?'} minutes

Findings:
{chr(10).join(findings_text) if findings_text else '- No detailed findings'}

Actions Taken:
{chr(10).join(f'- {action}' for action in inspection.actions_taken) if inspection.actions_taken else '- None'}

Recommendations:
{chr(10).join(f'- {rec}' for rec in inspection.recommendations) if inspection.recommendations else '- None'}

Notes: {findings.get('notes', inspection.notes or 'No additional notes')}
        """.strip()

        return MemoryShard(
            id=f"inspection_{inspection.inspection_id}",
            text=text,
            metadata={
                "source": "keep_inspection",
                "type": "inspection",
                "hive_id": inspection.hive_id,
                "inspection_type": inspection.inspection_type.value,
                "timestamp": inspection.timestamp.isoformat(),
                "has_concerns": len(concerns) > 0,
                "concern_count": len(concerns),
            }
        )

    def _create_alert_shard(self, alert: Alert) -> Any:
        """Create memory shard for an alert."""
        from HoloLoom.documentation.types import MemoryShard

        hive = self.apiary.hives.get(alert.hive_id)
        hive_name = hive.name if hive else alert.hive_id

        text = f"""
ALERT: {alert.title}
Hive: {hive_name}
Level: {alert.level.value.upper()}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M')}

{alert.message}

Status: {'RESOLVED' if alert.resolved else 'ACTIVE'}
        """.strip()

        return MemoryShard(
            id=f"alert_{alert.alert_id}",
            text=text,
            metadata={
                "source": "keep_alert",
                "type": "alert",
                "hive_id": alert.hive_id,
                "alert_level": alert.level.value,
                "resolved": alert.resolved,
                "timestamp": alert.timestamp.isoformat(),
            }
        )


class HoloLoomQueryAdapter:
    """
    Adapts natural language queries to apiary-specific insights.

    Uses HoloLoom to answer beekeeping questions based on apiary state.
    """

    def __init__(self, apiary: Apiary):
        """
        Initialize query adapter.

        Args:
            apiary: Apiary to query
        """
        self.apiary = apiary
        self.memory_adapter = ApiaryMemoryAdapter(apiary)
        self._shuttle = None

    async def initialize(self) -> None:
        """Initialize HoloLoom shuttle with apiary memory."""
        try:
            from HoloLoom.weaving_shuttle import WeavingShuttle
            from HoloLoom.config import Config

            config = Config.fast()
            shards = self.memory_adapter.to_memory_shards()

            self._shuttle = WeavingShuttle(
                cfg=config,
                shards=shards,
                enable_reflection=True
            )

        except ImportError:
            raise RuntimeError("HoloLoom not available - cannot initialize query adapter")

    async def query(self, question: str) -> Dict[str, Any]:
        """
        Query apiary using natural language.

        Args:
            question: Natural language question

        Returns:
            Dict with 'answer', 'confidence', 'sources', etc.
        """
        if not self._shuttle:
            raise RuntimeError("Query adapter not initialized - call initialize() first")

        from HoloLoom.documentation.types import Query

        query = Query(text=question)
        spacetime = await self._shuttle.weave(query)

        return {
            "question": question,
            "answer": spacetime.response,
            "confidence": getattr(spacetime, "confidence", None),
            "sources": getattr(spacetime, "sources", []),
            "reasoning": getattr(spacetime, "reasoning", ""),
        }

    async def get_insights(self) -> Dict[str, Any]:
        """
        Get automated insights about the apiary.

        Returns:
            Dict with various insights and patterns detected
        """
        if not self._shuttle:
            raise RuntimeError("Query adapter not initialized")

        insights = {}

        # Query for patterns
        questions = [
            "What are the main health concerns across the apiary?",
            "Which colonies need immediate attention?",
            "What seasonal tasks should be prioritized?",
            "Are there any patterns in recent inspections?",
        ]

        for question in questions:
            result = await self.query(question)
            key = question.split("?")[0].lower().replace(" ", "_")
            insights[key] = result["answer"]

        return insights

    async def close(self) -> None:
        """Clean up resources."""
        if self._shuttle:
            await self._shuttle.close()


# =============================================================================
# Convenience Functions
# =============================================================================

async def create_hololoom_session(apiary: Apiary) -> HoloLoomQueryAdapter:
    """
    Create and initialize a HoloLoom query session.

    Args:
        apiary: Apiary to connect

    Returns:
        Initialized query adapter

    Example:
        async with create_hololoom_session(apiary) as session:
            result = await session.query("What needs attention?")
    """
    adapter = HoloLoomQueryAdapter(apiary)
    await adapter.initialize()
    return adapter


def export_to_memory(apiary: Apiary) -> List[Any]:
    """
    Export apiary to HoloLoom memory shards.

    Args:
        apiary: Apiary to export

    Returns:
        List of MemoryShard objects

    Example:
        shards = export_to_memory(apiary)
        # Use shards with HoloLoom shuttle
    """
    adapter = ApiaryMemoryAdapter(apiary)
    return adapter.to_memory_shards()
