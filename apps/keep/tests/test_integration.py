"""
Integration tests for Keep beekeeping application.

Tests complete workflows across multiple components:
- Apiary management workflows
- Inspection → Alert generation
- Analytics integration
- Journal integration
- Builder → Validator → Storage workflows
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from apps.keep import (
    # Core
    Apiary,
    BeeKeeper,
    # Builders
    hive,
    colony,
    inspection,
    # Types
    HealthStatus,
    QueenStatus,
    InspectionType,
    # Analytics
    ApiaryAnalytics,
    quick_health_check,
    # Journal
    create_journal,
    EntryType,
    # Transforms
    filter_healthy,
    get_top_healthy_colonies,
    # Validation
)
from apps.keep.validation import (
    ColonyValidator,
    InspectionValidator,
    assert_valid_colony,
)


class TestApiaryWorkflows:
    """Test complete apiary management workflows."""

    def test_complete_hive_lifecycle(self):
        """Test complete hive lifecycle from creation to harvest."""
        # Setup
        apiary = Apiary(name="Test Apiary")

        # Create hive
        h = (hive("Test Hive")
            .langstroth()
            .at("Test Location")
            .installed_on(datetime(2024, 4, 1))
            .build())

        hive_id = apiary.add_hive(h)
        assert hive_id in apiary.hives

        # Add colony
        c = (colony()
            .in_hive(hive_id)
            .italian()
            .from_package()
            .healthy()
            .queen_laying()
            .population(50000)
            .build())

        colony_id = apiary.add_colony(c)
        assert colony_id in apiary.colonies

        # Record inspection
        i = (inspection()
            .for_hive(hive_id)
            .colony(colony_id)
            .routine()
            .on(datetime.now())
            .queen_seen()
            .eggs_present()
            .brood_frames(8)
            .honey_frames(6)
            .no_pests()
            .build())

        inspection_id = apiary.record_inspection(i)
        assert len(apiary.inspections) == 1

        # Verify colony was updated
        updated_colony = apiary.colonies[colony_id]
        assert updated_colony.queen_status == QueenStatus.PRESENT_LAYING

        # Check summary
        summary = apiary.get_apiary_summary()
        assert summary['total_hives'] == 1
        assert summary['active_colonies'] == 1
        assert summary['inspections_recorded'] == 1

    def test_inspection_generates_alerts(self):
        """Test that inspections generate appropriate alerts."""
        apiary = Apiary(name="Test Apiary")

        # Setup
        h = hive("Test").langstroth().build()
        hive_id = apiary.add_hive(h)

        c = colony().in_hive(hive_id).healthy().build()
        colony_id = apiary.add_colony(c)

        # Record concerning inspection
        i = (inspection()
            .for_hive(hive_id)
            .colony(colony_id)
            .health_check()
            .on(datetime.now())
            .queen_seen(False)
            .eggs_present(False)
            .mites()  # Mites detected
            .build())

        apiary.record_inspection(i)

        # Check alerts were generated
        alerts = apiary.get_active_alerts()
        assert len(alerts) > 0

        # Should have alerts for both issues
        alert_titles = [a.title for a in alerts]
        assert any("mites" in title.lower() for title in alert_titles)

    def test_alert_resolution_workflow(self):
        """Test alert creation and resolution workflow."""
        apiary = Apiary(name="Test Apiary")

        h = hive("Test").langstroth().build()
        hive_id = apiary.add_hive(h)

        c = colony().in_hive(hive_id).build()
        colony_id = apiary.add_colony(c)

        # Create inspection that generates alerts
        i = (inspection()
            .for_hive(hive_id)
            .colony(colony_id)
            .health_check()
            .mites()
            .build())

        apiary.record_inspection(i)

        # Get alerts
        alerts = apiary.get_active_alerts(hive_id)
        assert len(alerts) > 0

        # Resolve first alert
        alert_id = alerts[0].alert_id
        resolved = apiary.resolve_alert(alert_id)
        assert resolved

        # Verify resolution
        remaining = apiary.get_active_alerts(hive_id)
        assert len(remaining) < len(alerts)


class TestAnalyticsIntegration:
    """Test analytics integration with apiary data."""

    def test_analytics_with_real_data(self):
        """Test analytics on realistic apiary data."""
        apiary = Apiary(name="Analytics Test")

        # Create multiple hives with varying health
        for i in range(5):
            h = hive(f"Hive {i}").langstroth().build()
            hive_id = apiary.add_hive(h)

            health = [
                HealthStatus.EXCELLENT,
                HealthStatus.GOOD,
                HealthStatus.FAIR,
                HealthStatus.POOR,
                HealthStatus.CRITICAL
            ][i]

            c = (colony()
                .in_hive(hive_id)
                .health(health)
                .population(50000 - i * 10000)
                .build())

            apiary.add_colony(c)

        # Run analytics
        analytics = ApiaryAnalytics(apiary)

        # Health score
        health_score = analytics.compute_health_score()
        assert 0 <= health_score['score'] <= 100
        assert health_score['grade'] in ['A', 'B', 'C', 'D', 'F']
        assert 'distribution' in health_score

        # Risk assessment
        risk = analytics.assess_risk()
        assert risk.overall_risk in ['low', 'medium', 'high', 'critical']
        assert len(risk.risk_factors) > 0

        # Quick check
        quick = quick_health_check(apiary)
        assert 'health_grade' in quick
        assert 'risk_level' in quick

    def test_colony_comparison(self):
        """Test colony comparison functionality."""
        apiary = Apiary(name="Comparison Test")

        # Create colonies with different characteristics
        for i in range(3):
            h = hive(f"Hive {i}").langstroth().build()
            hive_id = apiary.add_hive(h)

            c = (colony()
                .in_hive(hive_id)
                .population(30000 + i * 10000)
                .queen_age(6 + i * 6)
                .build())

            apiary.add_colony(c)

        # Run comparison
        analytics = ApiaryAnalytics(apiary)
        comparisons = analytics.compare_colonies()

        assert len(comparisons) == 3
        assert all('population' in comp for comp in comparisons)
        assert all('age_days' in comp for comp in comparisons)


class TestJournalIntegration:
    """Test journal integration workflows."""

    def test_journal_with_apiary(self):
        """Test journal creation and integration."""
        apiary = Apiary(name="Journal Test")

        h = hive("Test Hive").langstroth().build()
        hive_id = apiary.add_hive(h)

        # Create journal
        journal = create_journal(apiary)
        assert journal.apiary is apiary

        # Record entries
        journal.observe(
            "First observation",
            hive_ids=[hive_id],
            tags=["observation"]
        )

        journal.celebrate(
            "First milestone!",
            hive_ids=[hive_id],
            tags=["milestone"]
        )

        assert len(journal.entries) == 2

        # Get entries
        hive_entries = journal.get_entries(hive_id=hive_id)
        assert len(hive_entries) == 2

    @pytest.mark.asyncio
    async def test_journal_insights(self):
        """Test journal insights extraction."""
        apiary = Apiary(name="Insights Test")

        h = hive("Test").langstroth().build()
        hive_id = apiary.add_hive(h)

        journal = create_journal(apiary)

        # Add multiple entries
        for i in range(5):
            journal.observe(
                f"Observation {i}",
                hive_ids=[hive_id],
                tags=["routine"]
            )

        # Extract insights
        insights = await journal.extract_insights()

        assert 'insights' in insights
        assert 'patterns' in insights
        assert 'recommendations' in insights


class TestBuilderValidationIntegration:
    """Test integration of builders with validation."""

    def test_builder_produces_valid_objects(self):
        """Test that builders produce valid objects."""
        # Build colony
        c = (colony()
            .in_hive("test_hive")
            .italian()
            .healthy()
            .population(50000)
            .build())

        # Validate
        errors = ColonyValidator.validate(c)
        assert len(errors) == 0

    def test_builder_with_invalid_data(self):
        """Test builder can create invalid objects (validation is separate)."""
        # Create colony with invalid data
        c = (colony()
            .in_hive("test_hive")
            .population(-1000)  # Invalid
            .build())

        # Builder allows it (validation catches it)
        assert c.population_estimate == -1000

        # But validation catches it
        errors = ColonyValidator.validate(c)
        assert len(errors) > 0
        assert any("negative" in err.lower() for err in errors)

    def test_validation_in_apiary_workflow(self):
        """Test that validation can be integrated into workflows."""
        apiary = Apiary(name="Validation Test")

        h = hive("Test").langstroth().build()
        hive_id = apiary.add_hive(h)

        # Create valid colony
        c = (colony()
            .in_hive(hive_id)
            .population(50000)
            .build())

        # Validate before adding
        assert_valid_colony(c)  # Should not raise

        # Add to apiary
        colony_id = apiary.add_colony(c)
        assert colony_id in apiary.colonies


class TestTransformIntegration:
    """Test functional transforms with real data."""

    def test_transform_pipeline_on_apiary(self):
        """Test complete transform pipeline."""
        apiary = Apiary(name="Transform Test")

        # Create varied colonies
        for i in range(10):
            h = hive(f"Hive {i}").langstroth().build()
            hive_id = apiary.add_hive(h)

            health = [HealthStatus.EXCELLENT, HealthStatus.GOOD, HealthStatus.FAIR][i % 3]
            pop = 30000 + (i * 5000)

            c = (colony()
                .in_hive(hive_id)
                .health(health)
                .population(pop)
                .build())

            apiary.add_colony(c)

        # Apply transforms
        all_colonies = list(apiary.colonies.values())

        # Get top 3 healthy
        top_3 = get_top_healthy_colonies(3)(all_colonies)

        assert len(top_3) == 3
        assert all(
            c.health_status in [HealthStatus.EXCELLENT, HealthStatus.GOOD]
            for c in top_3
        )

        # Should be sorted by population (descending)
        for i in range(len(top_3) - 1):
            assert top_3[i].population_estimate >= top_3[i + 1].population_estimate

    def test_filter_integration(self):
        """Test filters on real apiary data."""
        apiary = Apiary(name="Filter Test")

        # Create colonies
        for i in range(5):
            h = hive(f"Hive {i}").langstroth().build()
            hive_id = apiary.add_hive(h)

            if i < 2:
                c = (colony()
                    .in_hive(hive_id)
                    .excellent_health()
                    .population(50000)
                    .build())
            else:
                c = (colony()
                    .in_hive(hive_id)
                    .poor_health()
                    .population(50000)
                    .build())

            apiary.add_colony(c)

        colonies = list(apiary.colonies.values())

        # Filter healthy
        healthy = filter_healthy(colonies)
        assert len(healthy) == 2


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    @pytest.mark.asyncio
    async def test_complete_beekeeping_season(self):
        """Test simulation of a complete beekeeping season."""
        # Setup apiary
        apiary = Apiary(name="Seasonal Test", location="Test Location")

        # Spring: Add hives and colonies
        hive1 = hive("Alpha").langstroth().at("East field").build()
        hive_id_1 = apiary.add_hive(hive1)

        colony1 = (colony()
            .in_hive(hive_id_1)
            .italian()
            .from_package()
            .excellent_health()
            .queen_laying()
            .population(50000)
            .build())

        colony_id_1 = apiary.add_colony(colony1)

        # Spring inspection
        spring_insp = (inspection()
            .for_hive(hive_id_1)
            .colony(colony_id_1)
            .routine()
            .on(datetime(2024, 4, 15))
            .queen_seen()
            .eggs_present()
            .brood_frames(6)
            .build())

        apiary.record_inspection(spring_insp)

        # Summer: Growth and expansion
        summer_insp = (inspection()
            .for_hive(hive_id_1)
            .colony(colony_id_1)
            .routine()
            .on(datetime(2024, 7, 15))
            .queen_seen()
            .eggs_present()
            .brood_frames(8)
            .honey_frames(8)
            .population(60000)
            .build())

        apiary.record_inspection(summer_insp)

        # Harvest
        from apps.keep.models import HarvestRecord
        harvest = HarvestRecord(
            hive_id=hive_id_1,
            timestamp=datetime(2024, 8, 1),
            product_type="honey",
            quantity=45.0,
            unit="lbs"
        )
        apiary.record_harvest(harvest)

        # Analytics
        analytics = ApiaryAnalytics(apiary)
        health = analytics.compute_health_score()
        assert health['score'] > 70  # Should be healthy

        # Journal
        journal = create_journal(apiary)
        journal.celebrate(
            "First honey harvest! 45 lbs from Alpha.",
            hive_ids=[hive_id_1],
            tags=["harvest", "milestone"]
        )

        # BeeKeeper recommendations
        keeper = BeeKeeper(apiary)
        recommendations = await keeper.get_recommendations()

        assert len(recommendations) > 0

        # Final summary
        summary = apiary.get_apiary_summary()
        assert summary['total_hives'] == 1
        assert summary['active_colonies'] == 1
        assert summary['inspections_recorded'] == 2
        assert summary['yearly_harvest_lbs'] == 45.0


class TestErrorHandling:
    """Test error handling across integrations."""

    def test_invalid_colony_addition(self):
        """Test adding colony to non-existent hive."""
        apiary = Apiary(name="Error Test")

        c = colony().in_hive("nonexistent").build()

        with pytest.raises(ValueError):
            apiary.add_colony(c)

    def test_invalid_inspection(self):
        """Test recording inspection for non-existent hive."""
        apiary = Apiary(name="Error Test")

        i = inspection().for_hive("nonexistent").build()

        with pytest.raises(ValueError):
            apiary.record_inspection(i)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
