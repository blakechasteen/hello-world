"""
Unit tests for fluent builders.

Tests all builder patterns for correctness, type safety, and ergonomics.
"""

import pytest
from datetime import datetime

from apps.keep.builders import (
    HiveBuilder,
    ColonyBuilder,
    InspectionBuilder,
    AlertBuilder,
    hive,
    colony,
    inspection,
    alert,
)
from apps.keep.types import (
    HiveType,
    HealthStatus,
    QueenStatus,
    InspectionType,
    AlertLevel,
)


class TestHiveBuilder:
    """Test HiveBuilder fluent interface."""

    def test_basic_hive_creation(self):
        """Test basic hive creation with defaults."""
        h = HiveBuilder().build()
        assert h.hive_id is not None
        assert h.name == ""
        assert h.hive_type == HiveType.LANGSTROTH

    def test_fluent_chain(self):
        """Test fluent method chaining."""
        h = (HiveBuilder()
            .named("Test Hive")
            .langstroth()
            .at("Test Location")
            .notes("Test notes")
            .build())

        assert h.name == "Test Hive"
        assert h.hive_type == HiveType.LANGSTROTH
        assert h.location == "Test Location"
        assert h.notes == "Test notes"

    def test_all_hive_types(self):
        """Test all hive type setters."""
        assert HiveBuilder().langstroth().build().hive_type == HiveType.LANGSTROTH
        assert HiveBuilder().top_bar().build().hive_type == HiveType.TOP_BAR
        assert HiveBuilder().warre().build().hive_type == HiveType.WARRE
        assert HiveBuilder().flow_hive().build().hive_type == HiveType.FLOW_HIVE
        assert HiveBuilder().observation().build().hive_type == HiveType.OBSERVATION

    def test_metadata(self):
        """Test metadata addition."""
        h = (HiveBuilder()
            .metadata(color="white", frames=10)
            .build())

        assert h.metadata["color"] == "white"
        assert h.metadata["frames"] == 10

    def test_convenience_function(self):
        """Test convenience hive() function."""
        h = hive("Alpha").langstroth().build()
        assert h.name == "Alpha"

    def test_installation_date(self):
        """Test installation date setting."""
        date = datetime(2024, 4, 1)
        h = HiveBuilder().installed_on(date).build()
        assert h.installation_date == date

    def test_immutability_of_build(self):
        """Test that build() returns new instance each time."""
        builder = HiveBuilder().named("Test")
        h1 = builder.build()
        h2 = builder.build()

        # Should be different objects with same data
        assert h1 is not h2
        assert h1.name == h2.name


class TestColonyBuilder:
    """Test ColonyBuilder fluent interface."""

    def test_basic_colony_creation(self):
        """Test basic colony creation."""
        c = ColonyBuilder().build()
        assert c.colony_id is not None
        assert c.health_status == HealthStatus.GOOD

    def test_breed_setters(self):
        """Test breed convenience methods."""
        assert ColonyBuilder().italian().build().breed == "Italian"
        assert ColonyBuilder().carniolan().build().breed == "Carniolan"
        assert ColonyBuilder().russian().build().breed == "Russian"
        assert ColonyBuilder().buckfast().build().breed == "Buckfast"

    def test_origin_setters(self):
        """Test origin convenience methods."""
        assert ColonyBuilder().from_package().build().origin == "package"
        assert ColonyBuilder().from_nuc().build().origin == "nuc"
        assert ColonyBuilder().from_swarm().build().origin == "swarm"
        assert ColonyBuilder().from_split().build().origin == "split"
        assert ColonyBuilder().from_split("Hive001").build().origin == "split from Hive001"

    def test_health_setters(self):
        """Test health status convenience methods."""
        assert ColonyBuilder().healthy().build().health_status == HealthStatus.GOOD
        assert ColonyBuilder().excellent_health().build().health_status == HealthStatus.EXCELLENT
        assert ColonyBuilder().fair_health().build().health_status == HealthStatus.FAIR
        assert ColonyBuilder().poor_health().build().health_status == HealthStatus.POOR

    def test_queen_status_setters(self):
        """Test queen status convenience methods."""
        assert ColonyBuilder().queen_laying().build().queen_status == QueenStatus.PRESENT_LAYING
        assert ColonyBuilder().queen_not_laying().build().queen_status == QueenStatus.PRESENT_NOT_LAYING
        assert ColonyBuilder().queenless().build().queen_status == QueenStatus.ABSENT

    def test_full_colony_chain(self):
        """Test complete fluent chain."""
        c = (colony()
            .in_hive("hive123")
            .italian()
            .from_package()
            .excellent_health()
            .queen_laying()
            .population(50000)
            .queen_age(6)
            .notes("Test colony")
            .build())

        assert c.hive_id == "hive123"
        assert c.breed == "Italian"
        assert c.origin == "package"
        assert c.health_status == HealthStatus.EXCELLENT
        assert c.queen_status == QueenStatus.PRESENT_LAYING
        assert c.population_estimate == 50000
        assert c.queen_age_months == 6
        assert c.notes == "Test colony"


class TestInspectionBuilder:
    """Test InspectionBuilder fluent interface."""

    def test_basic_inspection_creation(self):
        """Test basic inspection creation."""
        i = InspectionBuilder().build()
        assert i.inspection_id is not None
        assert i.inspection_type == InspectionType.ROUTINE

    def test_inspection_type_setters(self):
        """Test inspection type convenience methods."""
        assert InspectionBuilder().routine().build().inspection_type == InspectionType.ROUTINE
        assert InspectionBuilder().health_check().build().inspection_type == InspectionType.HEALTH_CHECK
        assert InspectionBuilder().swarm_check().build().inspection_type == InspectionType.SWARM_CHECK
        assert InspectionBuilder().harvest().build().inspection_type == InspectionType.HARVEST

    def test_findings_builders(self):
        """Test findings construction."""
        i = (inspection()
            .queen_seen()
            .eggs_present()
            .larvae_present()
            .capped_brood()
            .brood_frames(8)
            .honey_frames(6)
            .pollen_frames(3)
            .population(50000)
            .build())

        assert i.findings["queen_seen"] is True
        assert i.findings["eggs_seen"] is True
        assert i.findings["larvae_seen"] is True
        assert i.findings["capped_brood_seen"] is True
        assert i.findings["frames_with_brood"] == 8
        assert i.findings["frames_with_honey"] == 6
        assert i.findings["frames_with_pollen"] == 3
        assert i.findings["population_estimate"] == 50000

    def test_pest_findings(self):
        """Test pest observation builders."""
        i1 = inspection().mites().build()
        assert i1.findings["mites_observed"] is True

        i2 = inspection().beetles().build()
        assert i2.findings["beetles_observed"] is True

        i3 = inspection().no_pests().build()
        assert i3.findings["mites_observed"] is False
        assert i3.findings["beetles_observed"] is False
        assert i3.findings["moths_observed"] is False

    def test_actions_and_recommendations(self):
        """Test actions and recommendations."""
        i = (inspection()
            .action("Added super")
            .action("Checked frames")
            .recommend("Monitor closely")
            .build())

        assert "Added super" in i.actions_taken
        assert "Checked frames" in i.actions_taken
        assert "Monitor closely" in i.recommendations

    def test_custom_findings(self):
        """Test custom finding addition."""
        i = (inspection()
            .finding("custom_metric", 42)
            .findings(temperature=72.0, humidity=65)
            .build())

        assert i.findings["custom_metric"] == 42
        assert i.findings["temperature"] == 72.0
        assert i.findings["humidity"] == 65

    def test_complete_inspection_chain(self):
        """Test complete inspection workflow."""
        i = (inspection()
            .for_hive("hive123")
            .colony("colony456")
            .routine()
            .on(datetime(2024, 10, 28))
            .weather("Sunny")
            .temperature(72.0)
            .queen_seen()
            .eggs_present()
            .brood_frames(8)
            .no_pests()
            .action("Checked all frames")
            .inspector("John Doe")
            .duration(30)
            .build())

        assert i.hive_id == "hive123"
        assert i.colony_id == "colony456"
        assert i.inspection_type == InspectionType.ROUTINE
        assert i.weather == "Sunny"
        assert i.temperature == 72.0
        assert i.inspector == "John Doe"
        assert i.duration_minutes == 30


class TestAlertBuilder:
    """Test AlertBuilder fluent interface."""

    def test_basic_alert_creation(self):
        """Test basic alert creation."""
        a = AlertBuilder().build()
        assert a.alert_id is not None
        assert a.level == AlertLevel.INFO

    def test_alert_level_setters(self):
        """Test alert level convenience methods."""
        assert AlertBuilder().info().build().level == AlertLevel.INFO
        assert AlertBuilder().warning().build().level == AlertLevel.WARNING
        assert AlertBuilder().urgent().build().level == AlertLevel.URGENT
        assert AlertBuilder().critical().build().level == AlertLevel.CRITICAL

    def test_complete_alert_chain(self):
        """Test complete alert workflow."""
        a = (alert()
            .for_hive("hive123")
            .colony("colony456")
            .critical()
            .titled("Queenless colony")
            .message("No queen or eggs detected")
            .metadata(inspector="John Doe")
            .build())

        assert a.hive_id == "hive123"
        assert a.colony_id == "colony456"
        assert a.level == AlertLevel.CRITICAL
        assert a.title == "Queenless colony"
        assert a.message == "No queen or eggs detected"
        assert a.metadata["inspector"] == "John Doe"


class TestBuilderEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_builds(self):
        """Test that empty builds work with defaults."""
        assert HiveBuilder().build() is not None
        assert ColonyBuilder().build() is not None
        assert InspectionBuilder().build() is not None
        assert AlertBuilder().build() is not None

    def test_repeated_builds(self):
        """Test that builder can be reused."""
        builder = hive("Test")
        h1 = builder.build()
        h2 = builder.build()

        # Should create different instances
        assert h1.hive_id != h2.hive_id
        assert h1.name == h2.name

    def test_overriding_methods(self):
        """Test that later calls override earlier ones."""
        h = (hive("First")
            .named("Second")
            .named("Third")
            .build())

        assert h.name == "Third"

    def test_metadata_accumulation(self):
        """Test that metadata accumulates."""
        h = (hive("Test")
            .metadata(a=1)
            .metadata(b=2)
            .build())

        assert h.metadata["a"] == 1
        assert h.metadata["b"] == 2


# Property-based tests (if hypothesis installed)
try:
    from hypothesis import given, strategies as st

    class TestBuilderProperties:
        """Property-based tests for builders."""

        @given(st.text(min_size=1, max_size=100))
        def test_hive_name_property(self, name):
            """Property: Any valid string can be a hive name."""
            h = hive(name).build()
            assert h.name == name

        @given(st.integers(min_value=0, max_value=1000000))
        def test_colony_population_property(self, pop):
            """Property: Any non-negative integer is valid population."""
            c = colony().population(pop).build()
            assert c.population_estimate == pop

        @given(st.integers(min_value=0, max_value=10))
        def test_inspection_frames_property(self, frames):
            """Property: Frame counts are preserved."""
            i = (inspection()
                .brood_frames(frames)
                .honey_frames(frames)
                .build())

            assert i.findings["frames_with_brood"] == frames
            assert i.findings["frames_with_honey"] == frames

except ImportError:
    # Hypothesis not installed, skip property tests
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
