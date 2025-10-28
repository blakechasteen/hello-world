"""
Protocol definitions for Keep beekeeping application.

Defines extensible protocols for data sources, analyzers, and integrations,
enabling elegant composition and dependency injection following mythRL patterns.
"""

from typing import Protocol, List, Dict, Any, Optional, AsyncIterator
from datetime import datetime

from apps.keep.models import Hive, Colony, Inspection, HarvestRecord, Alert
from apps.keep.types import InspectionData, HealthStatus, QueenStatus, AlertLevel


class InspectionDataSource(Protocol):
    """
    Protocol for sources of inspection data.

    Enables integration with various data collection methods:
    - Manual entry forms
    - Mobile apps
    - IoT sensors (hive scales, temperature monitors)
    - Voice recording transcriptions
    - External APIs
    """

    async def fetch_inspections(
        self,
        hive_id: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[Inspection]:
        """Fetch inspection records from this data source."""
        ...

    async def submit_inspection(self, inspection: Inspection) -> str:
        """Submit a new inspection to this data source."""
        ...


class ColonyHealthAnalyzer(Protocol):
    """
    Protocol for colony health analysis algorithms.

    Enables pluggable health assessment strategies:
    - Rule-based scoring
    - ML-based prediction
    - Expert system rules
    - Historical trend analysis
    """

    def analyze_health(
        self,
        colony: Colony,
        recent_inspections: List[Inspection],
        context: Dict[str, Any]
    ) -> HealthStatus:
        """Analyze colony health and return status."""
        ...

    def get_health_factors(
        self,
        colony: Colony,
        recent_inspections: List[Inspection]
    ) -> Dict[str, float]:
        """Return contributing factors to health assessment."""
        ...


class AlertGenerator(Protocol):
    """
    Protocol for alert generation strategies.

    Enables customizable alerting logic:
    - Threshold-based alerts
    - Trend-based predictions
    - ML anomaly detection
    - Expert knowledge rules
    """

    def generate_alerts(
        self,
        inspection: Inspection,
        colony: Colony,
        history: List[Inspection]
    ) -> List[Alert]:
        """Generate alerts based on inspection findings."""
        ...

    def assess_urgency(
        self,
        alert_type: str,
        context: Dict[str, Any]
    ) -> AlertLevel:
        """Determine urgency level for an alert type."""
        ...


class RecommendationEngine(Protocol):
    """
    Protocol for recommendation generation.

    Enables diverse recommendation strategies:
    - Rule-based recommendations
    - ML-powered suggestions
    - Expert system advice
    - Community best practices
    """

    async def get_recommendations(
        self,
        apiary_state: Dict[str, Any],
        planning_horizon_days: int
    ) -> List[Any]:  # Returns KeeperRecommendation objects
        """Generate prioritized recommendations."""
        ...


class ApiaryStateExporter(Protocol):
    """
    Protocol for exporting apiary state to various formats.

    Enables integration with:
    - Beekeeping association formats
    - Spreadsheet exports
    - Backup systems
    - Analytics platforms
    - HoloLoom memory systems
    """

    def export_state(
        self,
        apiary: Any,  # Apiary type
        format: str
    ) -> bytes:
        """Export apiary state in specified format."""
        ...

    def export_to_memory_shards(
        self,
        apiary: Any
    ) -> List[Any]:  # Returns MemoryShard objects
        """Export apiary state as HoloLoom memory shards."""
        ...


class WeatherDataProvider(Protocol):
    """
    Protocol for weather data integration.

    Enables integration with weather services for:
    - Inspection timing recommendations
    - Seasonal forecasts
    - Historical weather correlation
    """

    async def get_current_weather(self, location: str) -> Dict[str, Any]:
        """Get current weather conditions."""
        ...

    async def get_forecast(
        self,
        location: str,
        days: int
    ) -> List[Dict[str, Any]]:
        """Get weather forecast."""
        ...

    def is_suitable_for_inspection(
        self,
        weather: Dict[str, Any]
    ) -> bool:
        """Determine if weather is suitable for hive inspection."""
        ...


class JournalIntegration(Protocol):
    """
    Protocol for narrative journal integration.

    Enables rich narrative tracking like food_e journal:
    - Natural language inspection notes
    - Temporal narrative synthesis
    - Pattern recognition across entries
    - Emotional/sentiment tracking
    """

    async def record_entry(
        self,
        entry_type: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Record a journal entry."""
        ...

    async def synthesize_narrative(
        self,
        hive_id: str,
        time_range: tuple[datetime, datetime]
    ) -> str:
        """Generate narrative summary for a time period."""
        ...

    async def extract_insights(
        self,
        entries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract insights from journal entries."""
        ...