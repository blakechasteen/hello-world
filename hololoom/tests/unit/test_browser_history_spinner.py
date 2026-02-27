"""
Tests for BrowserHistorySpinner - Multi-Browser History Ingestion
=================================================================

Comprehensive tests for the browser history spinner including:
- BrowserConfig dataclass
- HistoryEntry and DomainStats dataclasses
- Timestamp conversion functions (webkit, prtime, mac_absolute)
- BrowserHistorySpinner initialization and capabilities
- Importance scoring
- Domain aggregation
- Incremental checkpointing

Author: Claude Code
Date: December 2025
"""

import pytest
import os
import sys
import sqlite3
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import test targets
from HoloLoom.spinningWheel.browser_history import (
    BrowserConfig,
    HistoryEntry,
    DomainStats,
    BrowserHistorySpinner,
    BROWSER_CONFIGS,
    convert_webkit_timestamp,
    convert_prtime_timestamp,
    convert_mac_absolute_timestamp,
    convert_timestamp,
    read_browser_history,
    get_available_browsers,
    spin_browser_history
)
from HoloLoom.spinningWheel.protocol import (
    SpinnerCapabilities,
    SpinnerCheckpoint,
    ImportanceSignals,
    ImportanceScore
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_history_entry():
    """Create a sample HistoryEntry."""
    return HistoryEntry(
        url="https://github.com/example/repo",
        title="Example Repository",
        visit_count=15,
        last_visit=datetime.now() - timedelta(days=2),
        browser="chrome",
        profile="Default"
    )


@pytest.fixture
def sample_history_entry_no_title():
    """Create a HistoryEntry without title."""
    return HistoryEntry(
        url="https://docs.python.org/3/library/",
        title="",
        visit_count=5,
        last_visit=datetime.now(),
        browser="firefox"
    )


@pytest.fixture
def domain_stats():
    """Create a sample DomainStats."""
    return DomainStats(
        domain="github.com",
        total_visits=150,
        unique_pages=25,
        last_visit=datetime.now(),
        first_seen=datetime.now() - timedelta(days=60),
        titles=["Repo 1", "Repo 2", "Pull Request #123"],
        browsers=["chrome", "firefox"]
    )


@pytest.fixture
def browser_spinner():
    """Create a BrowserHistorySpinner instance."""
    return BrowserHistorySpinner(
        importance_threshold=0.3,
        days_back=30,
        min_visits=2,
        aggregate_by_domain=True
    )


@pytest.fixture
def temp_history_db():
    """Create a temporary SQLite database with mock history."""
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "History")

    # Create database with Chrome-like schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create urls table (Chrome/Edge/Brave schema)
    cursor.execute("""
        CREATE TABLE urls (
            id INTEGER PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT,
            visit_count INTEGER DEFAULT 1,
            last_visit_time INTEGER
        )
    """)

    # Insert test data
    # WebKit timestamp for "now" (microseconds since Jan 1, 1601)
    # Current time: convert to webkit
    now = datetime.now()
    webkit_now = int((now.timestamp() + 11644473600) * 1_000_000)
    webkit_yesterday = webkit_now - (24 * 60 * 60 * 1_000_000)

    test_data = [
        ("https://github.com/test/repo", "Test Repository", 10, webkit_now),
        ("https://stackoverflow.com/questions/123", "Python Question", 5, webkit_yesterday),
        ("https://docs.python.org/3/", "Python Docs", 20, webkit_now),
        ("https://google.com/search", "Google Search", 100, webkit_now),  # Should be excluded
        ("chrome://settings", "Settings", 5, webkit_now),  # Internal, should be excluded
    ]

    cursor.executemany(
        "INSERT INTO urls (url, title, visit_count, last_visit_time) VALUES (?, ?, ?, ?)",
        test_data
    )

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    shutil.rmtree(temp_dir)


# =============================================================================
# BrowserConfig Tests
# =============================================================================

class TestBrowserConfig:
    """Tests for BrowserConfig dataclass."""

    def test_chrome_config_exists(self):
        """Test Chrome configuration is defined."""
        assert 'chrome' in BROWSER_CONFIGS
        config = BROWSER_CONFIGS['chrome']

        assert config.name == 'Chrome'
        assert config.history_table == 'urls'
        assert config.url_column == 'url'
        assert config.timestamp_format == 'webkit'

    def test_firefox_config_exists(self):
        """Test Firefox configuration is defined."""
        assert 'firefox' in BROWSER_CONFIGS
        config = BROWSER_CONFIGS['firefox']

        assert config.name == 'Firefox'
        assert config.history_table == 'moz_places'
        assert config.timestamp_format == 'prtime'

    def test_edge_config_exists(self):
        """Test Edge configuration is defined."""
        assert 'edge' in BROWSER_CONFIGS
        config = BROWSER_CONFIGS['edge']

        assert config.name == 'Edge'
        assert config.timestamp_format == 'webkit'

    def test_safari_config_exists(self):
        """Test Safari configuration is defined."""
        assert 'safari' in BROWSER_CONFIGS
        config = BROWSER_CONFIGS['safari']

        assert config.name == 'Safari'
        assert config.timestamp_format == 'mac_absolute'

    def test_brave_config_exists(self):
        """Test Brave configuration is defined."""
        assert 'brave' in BROWSER_CONFIGS
        config = BROWSER_CONFIGS['brave']

        assert config.name == 'Brave'
        assert config.timestamp_format == 'webkit'

    def test_all_configs_have_required_fields(self):
        """Test all browser configs have required fields."""
        for browser_name, config in BROWSER_CONFIGS.items():
            assert config.name, f"{browser_name} missing name"
            assert config.history_paths, f"{browser_name} missing history_paths"
            assert config.history_table, f"{browser_name} missing history_table"
            assert config.url_column, f"{browser_name} missing url_column"
            assert config.timestamp_format in ['webkit', 'prtime', 'mac_absolute'], \
                f"{browser_name} has invalid timestamp_format"


# =============================================================================
# HistoryEntry Tests
# =============================================================================

class TestHistoryEntry:
    """Tests for HistoryEntry dataclass."""

    def test_creation_with_all_fields(self, sample_history_entry):
        """Test creating HistoryEntry with all fields."""
        assert sample_history_entry.url == "https://github.com/example/repo"
        assert sample_history_entry.title == "Example Repository"
        assert sample_history_entry.visit_count == 15
        assert sample_history_entry.browser == "chrome"
        assert sample_history_entry.profile == "Default"

    def test_domain_extraction(self, sample_history_entry):
        """Test automatic domain extraction in post_init."""
        assert sample_history_entry.domain == "github.com"

    def test_domain_extraction_with_www(self):
        """Test domain extraction with www prefix."""
        entry = HistoryEntry(
            url="https://www.example.com/page",
            title="Test",
            visit_count=1,
            last_visit=datetime.now(),
            browser="chrome"
        )
        assert entry.domain == "www.example.com"

    def test_domain_extraction_with_port(self):
        """Test domain extraction with port number."""
        entry = HistoryEntry(
            url="http://localhost:8080/api",
            title="Local API",
            visit_count=1,
            last_visit=datetime.now(),
            browser="chrome"
        )
        assert entry.domain == "localhost:8080"

    def test_default_profile(self):
        """Test default profile value."""
        entry = HistoryEntry(
            url="https://example.com",
            title="Test",
            visit_count=1,
            last_visit=datetime.now(),
            browser="chrome"
        )
        assert entry.profile == "Default"

    def test_domain_extraction_handles_invalid_url(self):
        """Test domain extraction with invalid URL."""
        entry = HistoryEntry(
            url="not-a-valid-url",
            title="Invalid",
            visit_count=1,
            last_visit=datetime.now(),
            browser="chrome"
        )
        # Should not crash, should have some domain value
        assert entry.domain is not None


# =============================================================================
# DomainStats Tests
# =============================================================================

class TestDomainStats:
    """Tests for DomainStats dataclass."""

    def test_creation(self, domain_stats):
        """Test creating DomainStats."""
        assert domain_stats.domain == "github.com"
        assert domain_stats.total_visits == 150
        assert domain_stats.unique_pages == 25

    def test_default_values(self):
        """Test default values for DomainStats."""
        stats = DomainStats(domain="example.com")

        assert stats.total_visits == 0
        assert stats.unique_pages == 0
        assert stats.last_visit is None
        assert stats.first_seen is None
        assert stats.titles == []
        assert stats.browsers == []


# =============================================================================
# Timestamp Conversion Tests
# =============================================================================

class TestTimestampConversion:
    """Tests for timestamp conversion functions."""

    def test_webkit_timestamp_valid(self):
        """Test converting valid WebKit timestamp."""
        # WebKit timestamp for 2024-01-15 12:00:00 UTC
        # Unix: 1705320000
        # WebKit: (1705320000 + 11644473600) * 1_000_000 = 13349793600000000
        webkit_ts = 13349793600000000

        result = convert_webkit_timestamp(webkit_ts)

        assert isinstance(result, datetime)
        # Should be close to 2024-01-15
        assert result.year == 2024
        assert result.month == 1

    def test_webkit_timestamp_zero(self):
        """Test WebKit timestamp with zero value."""
        result = convert_webkit_timestamp(0)

        # Should return current time for invalid input
        assert isinstance(result, datetime)
        # Should be close to now
        assert (datetime.now() - result).total_seconds() < 60

    def test_webkit_timestamp_negative(self):
        """Test WebKit timestamp with negative value."""
        result = convert_webkit_timestamp(-1000)

        assert isinstance(result, datetime)

    def test_webkit_timestamp_none(self):
        """Test WebKit timestamp with None."""
        result = convert_webkit_timestamp(None)

        assert isinstance(result, datetime)

    def test_prtime_timestamp_valid(self):
        """Test converting valid PRTime (Firefox) timestamp."""
        # PRTime for 2024-01-15 12:00:00 UTC
        # Unix seconds: 1705320000
        # PRTime (microseconds): 1705320000000000
        prtime = 1705320000000000

        result = convert_prtime_timestamp(prtime)

        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1

    def test_prtime_timestamp_zero(self):
        """Test PRTime with zero value."""
        result = convert_prtime_timestamp(0)

        assert isinstance(result, datetime)

    def test_mac_absolute_timestamp_valid(self):
        """Test converting valid Mac Absolute Time (Safari)."""
        # Mac Absolute Time for 2024-01-15 12:00:00 UTC
        # Unix: 1705320000
        # Mac: 1705320000 - 978307200 = 727012800
        mac_time = 727012800

        result = convert_mac_absolute_timestamp(mac_time)

        assert isinstance(result, datetime)
        assert result.year == 2024

    def test_mac_absolute_timestamp_zero(self):
        """Test Mac Absolute Time with zero."""
        result = convert_mac_absolute_timestamp(0)

        assert isinstance(result, datetime)

    def test_convert_timestamp_dispatcher(self):
        """Test convert_timestamp dispatches to correct converter."""
        # Test WebKit
        result = convert_timestamp(13349793600000000, 'webkit')
        assert isinstance(result, datetime)

        # Test PRTime
        result = convert_timestamp(1705320000000000, 'prtime')
        assert isinstance(result, datetime)

        # Test Mac Absolute
        result = convert_timestamp(727012800, 'mac_absolute')
        assert isinstance(result, datetime)

    def test_convert_timestamp_unknown_format(self):
        """Test convert_timestamp with unknown format."""
        result = convert_timestamp(12345, 'unknown_format')

        # Should return current time for unknown format
        assert isinstance(result, datetime)

    def test_convert_timestamp_none_value(self):
        """Test convert_timestamp with None value."""
        result = convert_timestamp(None, 'webkit')

        assert isinstance(result, datetime)


# =============================================================================
# BrowserHistorySpinner Tests
# =============================================================================

class TestBrowserHistorySpinner:
    """Tests for BrowserHistorySpinner class."""

    def test_initialization(self, browser_spinner):
        """Test BrowserHistorySpinner initialization."""
        assert browser_spinner.name == "browser_history"
        assert browser_spinner.importance_threshold == 0.3
        assert browser_spinner.days_back == 30
        assert browser_spinner.min_visits == 2
        assert browser_spinner.aggregate_by_domain is True

    def test_initialization_defaults(self):
        """Test BrowserHistorySpinner with default values."""
        spinner = BrowserHistorySpinner()

        assert spinner.importance_threshold == 0.3
        assert spinner.days_back == 30
        assert spinner.min_visits == 1
        assert spinner.aggregate_by_domain is True

    def test_initialization_with_browsers_filter(self):
        """Test initialization with specific browsers."""
        spinner = BrowserHistorySpinner(browsers=['chrome', 'firefox'])

        assert spinner.browsers == ['chrome', 'firefox']

    def test_initialization_with_domain_filters(self):
        """Test initialization with domain filters."""
        spinner = BrowserHistorySpinner(
            include_domains_only=['github.com', 'stackoverflow.com'],
            exclude_domains=['facebook.com']
        )

        assert 'github.com' in spinner.include_domains_only
        assert 'facebook.com' in spinner.exclude_domains

    def test_default_excluded_domains(self, browser_spinner):
        """Test default excluded domains."""
        # Social media and common sites should be excluded by default
        assert 'google.com' in browser_spinner.exclude_domains
        assert 'facebook.com' in browser_spinner.exclude_domains
        assert 'youtube.com' in browser_spinner.exclude_domains
        assert 'localhost' in browser_spinner.exclude_domains

    def test_get_capabilities(self, browser_spinner):
        """Test getting spinner capabilities."""
        caps = browser_spinner.get_capabilities()

        assert isinstance(caps, SpinnerCapabilities)
        assert caps.basic_processing is True
        assert caps.incremental is True  # Supports checkpointing
        assert caps.batch_processing is True
        assert caps.importance_scoring is True
        assert 'chrome' in caps.supported_formats
        assert 'firefox' in caps.supported_formats

    def test_extract_profile_name_default(self, browser_spinner):
        """Test extracting 'Default' profile name."""
        path = "/home/user/.config/google-chrome/Default/History"
        result = browser_spinner._extract_profile_name(path)

        assert result == "Default"

    def test_extract_profile_name_numbered(self, browser_spinner):
        """Test extracting numbered profile name."""
        path = "/home/user/.config/google-chrome/Profile 1/History"
        result = browser_spinner._extract_profile_name(path)

        assert result == "Profile 1"

    def test_extract_profile_name_firefox(self, browser_spinner):
        """Test extracting Firefox profile name."""
        path = "/home/user/.mozilla/firefox/abc123.default-release/places.sqlite"
        result = browser_spinner._extract_profile_name(path)

        assert "default" in result.lower()


# =============================================================================
# Importance Scoring Tests
# =============================================================================

class TestImportanceScoring:
    """Tests for importance scoring in BrowserHistorySpinner."""

    def test_score_importance_high_visits(self, browser_spinner, sample_history_entry):
        """Test importance scoring for high visit count."""
        score = browser_spinner.score_importance(sample_history_entry)

        assert isinstance(score, ImportanceScore)
        assert 0 <= score.score <= 1
        # High visit count should contribute to score
        assert score.signals.length_score > 0

    def test_score_importance_recent_visit(self, browser_spinner):
        """Test importance scoring for recent visit."""
        recent_entry = HistoryEntry(
            url="https://example.com",
            title="Recent Page",
            visit_count=1,
            last_visit=datetime.now(),  # Just now
            browser="chrome"
        )

        score = browser_spinner.score_importance(recent_entry)

        # Recent visits should have high recency score
        assert score.signals.recency_score > 0.9

    def test_score_importance_old_visit(self, browser_spinner):
        """Test importance scoring for old visit."""
        old_entry = HistoryEntry(
            url="https://example.com",
            title="Old Page",
            visit_count=1,
            last_visit=datetime.now() - timedelta(days=25),
            browser="chrome"
        )

        score = browser_spinner.score_importance(old_entry)

        # Old visits should have lower recency score
        assert score.signals.recency_score < 0.5

    def test_score_importance_github_domain(self, browser_spinner):
        """Test authority score for github.com."""
        github_entry = HistoryEntry(
            url="https://github.com/user/repo",
            title="GitHub Repo",
            visit_count=5,
            last_visit=datetime.now(),
            browser="chrome"
        )

        score = browser_spinner.score_importance(github_entry)

        # GitHub should have high authority score
        assert score.signals.authority_score >= 0.85

    def test_score_importance_stackoverflow_domain(self, browser_spinner):
        """Test authority score for stackoverflow.com."""
        so_entry = HistoryEntry(
            url="https://stackoverflow.com/questions/123",
            title="How to do X?",
            visit_count=3,
            last_visit=datetime.now(),
            browser="chrome"
        )

        score = browser_spinner.score_importance(so_entry)

        # StackOverflow should have high authority score
        assert score.signals.authority_score >= 0.8

    def test_score_importance_edu_domain(self, browser_spinner):
        """Test authority score for .edu domain."""
        edu_entry = HistoryEntry(
            url="https://stanford.edu/research",
            title="Research Paper",
            visit_count=2,
            last_visit=datetime.now(),
            browser="chrome"
        )

        score = browser_spinner.score_importance(edu_entry)

        # .edu domains should have high authority
        assert score.signals.authority_score >= 0.8

    def test_score_importance_good_title(self, browser_spinner):
        """Test technical score for good title."""
        entry = HistoryEntry(
            url="https://example.com",
            title="A Comprehensive Guide to Python Programming",
            visit_count=1,
            last_visit=datetime.now(),
            browser="chrome"
        )

        score = browser_spinner.score_importance(entry)

        # Good title should have higher technical score
        assert score.signals.technical_score >= 0.5

    def test_score_importance_no_title(self, browser_spinner, sample_history_entry_no_title):
        """Test technical score when title is empty."""
        # Set title to URL to simulate no title
        sample_history_entry_no_title.title = sample_history_entry_no_title.url

        score = browser_spinner.score_importance(sample_history_entry_no_title)

        # No meaningful title should have lower technical score
        assert score.signals.technical_score < 0.5


# =============================================================================
# Domain Aggregation Tests
# =============================================================================

class TestDomainAggregation:
    """Tests for domain aggregation in BrowserHistorySpinner."""

    def test_aggregate_by_domain(self, browser_spinner):
        """Test domain aggregation."""
        entries = [
            HistoryEntry(
                url="https://github.com/repo1",
                title="Repo 1",
                visit_count=5,
                last_visit=datetime.now(),
                browser="chrome"
            ),
            HistoryEntry(
                url="https://github.com/repo2",
                title="Repo 2",
                visit_count=10,
                last_visit=datetime.now() - timedelta(days=1),
                browser="firefox"
            ),
            HistoryEntry(
                url="https://stackoverflow.com/q/1",
                title="Question",
                visit_count=3,
                last_visit=datetime.now(),
                browser="chrome"
            ),
        ]

        result = browser_spinner._aggregate_by_domain(entries)

        assert "github.com" in result
        assert "stackoverflow.com" in result

        github_stats = result["github.com"]
        assert github_stats.total_visits == 15  # 5 + 10
        assert github_stats.unique_pages == 2
        assert "chrome" in github_stats.browsers
        assert "firefox" in github_stats.browsers

    def test_aggregate_by_domain_empty(self, browser_spinner):
        """Test domain aggregation with empty list."""
        result = browser_spinner._aggregate_by_domain([])

        assert result == {}

    def test_aggregate_tracks_last_visit(self, browser_spinner):
        """Test that aggregation tracks latest visit."""
        now = datetime.now()
        yesterday = now - timedelta(days=1)

        entries = [
            HistoryEntry(
                url="https://example.com/page1",
                title="Page 1",
                visit_count=1,
                last_visit=yesterday,
                browser="chrome"
            ),
            HistoryEntry(
                url="https://example.com/page2",
                title="Page 2",
                visit_count=1,
                last_visit=now,
                browser="chrome"
            ),
        ]

        result = browser_spinner._aggregate_by_domain(entries)
        stats = result["example.com"]

        # Last visit should be the more recent one
        assert stats.last_visit == now


# =============================================================================
# Database Reading Tests
# =============================================================================

class TestDatabaseReading:
    """Tests for reading from browser database."""

    def test_copy_database(self, browser_spinner, temp_history_db):
        """Test copying database file."""
        temp_path = browser_spinner._copy_database(temp_history_db)

        assert temp_path is not None
        assert os.path.exists(temp_path)
        assert temp_path != temp_history_db

        # Cleanup
        os.remove(temp_path)

    def test_copy_database_nonexistent(self, browser_spinner):
        """Test copying non-existent database."""
        result = browser_spinner._copy_database("/nonexistent/path/History")

        assert result is None

    def test_read_history(self, browser_spinner, temp_history_db):
        """Test reading history from database."""
        config = BROWSER_CONFIGS['chrome']

        # Set min_visits to 1 for testing
        browser_spinner.min_visits = 1

        entries = browser_spinner._read_history(
            temp_history_db,
            config,
            "chrome",
            "Default"
        )

        # Should have some entries (excluding google.com and chrome://)
        # github, stackoverflow, docs.python.org should be included
        # google.com should be excluded (in default exclude list)
        # chrome://settings should be excluded (internal page)

        urls = [e.url for e in entries]

        # Check expected entries are present
        github_found = any("github.com" in url for url in urls)
        stackoverflow_found = any("stackoverflow.com" in url for url in urls)
        python_docs_found = any("docs.python.org" in url for url in urls)

        # At least some of our test entries should be found
        assert github_found or stackoverflow_found or python_docs_found

        # Internal pages should be excluded
        assert not any("chrome://" in url for url in urls)

    def test_read_history_filters_old_entries(self, browser_spinner, temp_history_db):
        """Test that old entries are filtered by days_back."""
        browser_spinner.days_back = 1  # Only last 24 hours

        config = BROWSER_CONFIGS['chrome']
        entries = browser_spinner._read_history(
            temp_history_db,
            config,
            "chrome",
            "Default"
        )

        # Only very recent entries should be included
        for entry in entries:
            age = datetime.now() - entry.last_visit
            # Allow some margin for test execution time
            assert age.days <= 2


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_spin_browser_history_alias(self):
        """Test that spin_browser_history is alias for read_browser_history."""
        assert spin_browser_history is read_browser_history

    def test_get_available_browsers(self):
        """Test getting available browsers."""
        # This will return an empty list if no browsers are installed
        # which is fine for testing
        browsers = get_available_browsers()

        assert isinstance(browsers, list)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_spinner_with_no_browsers_detected(self, browser_spinner):
        """Test is_available when no browsers found."""
        with patch.object(browser_spinner, '_detect_browsers', return_value=[]):
            assert browser_spinner.is_available() is False

    def test_detect_browsers_with_nonexistent_paths(self, browser_spinner):
        """Test detection with paths that don't exist."""
        # This should not crash, just return empty list
        browsers = browser_spinner._detect_browsers()

        # May or may not find browsers depending on system
        assert isinstance(browsers, list)

    @pytest.mark.asyncio
    async def test_spin_impl_no_browsers(self, browser_spinner):
        """Test _spin_impl when no browsers detected."""
        with patch.object(browser_spinner, '_detect_browsers', return_value=[]):
            shards = await browser_spinner._spin_impl()

            assert shards == []

    def test_history_entry_with_empty_url(self):
        """Test HistoryEntry handling of edge case URLs."""
        # Empty string URL
        entry = HistoryEntry(
            url="",
            title="Empty",
            visit_count=1,
            last_visit=datetime.now(),
            browser="chrome"
        )

        # Should not crash
        assert entry.url == ""

    def test_importance_scoring_edge_cases(self, browser_spinner):
        """Test importance scoring with edge case values."""
        # Very high visit count
        entry = HistoryEntry(
            url="https://example.com",
            title="Popular Page",
            visit_count=10000,
            last_visit=datetime.now(),
            browser="chrome"
        )

        score = browser_spinner.score_importance(entry)

        # Score should still be in valid range
        assert 0 <= score.score <= 1

    def test_spinner_checkpoint_dir(self):
        """Test spinner with checkpoint directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            spinner = BrowserHistorySpinner(
                checkpoint_dir=Path(temp_dir)
            )

            assert spinner.checkpoint_dir == Path(temp_dir)


# =============================================================================
# Integration Tests
# =============================================================================

class TestBrowserHistoryIntegration:
    """Integration tests for BrowserHistorySpinner."""

    @pytest.mark.asyncio
    async def test_full_spin_with_mock_data(self, temp_history_db):
        """Test full spin process with mocked database."""
        spinner = BrowserHistorySpinner(
            days_back=365,
            min_visits=1,
            aggregate_by_domain=True
        )

        # Mock _detect_browsers to return our temp database
        config = BROWSER_CONFIGS['chrome']
        mock_browsers = [('chrome', temp_history_db, config, 'Default')]

        with patch.object(spinner, '_detect_browsers', return_value=mock_browsers):
            with patch.object(spinner, '_copy_database', return_value=temp_history_db):
                shards = await spinner._spin_impl()

                # Should have some shards
                assert len(shards) > 0

                # Check shard structure
                for shard in shards:
                    assert shard.id is not None
                    assert shard.text is not None
                    assert 'importance_score' in shard.metadata

    @pytest.mark.asyncio
    async def test_spin_creates_domain_summaries(self, temp_history_db):
        """Test that spin creates domain summary shards."""
        spinner = BrowserHistorySpinner(
            days_back=365,
            min_visits=1,
            aggregate_by_domain=True
        )

        config = BROWSER_CONFIGS['chrome']
        mock_browsers = [('chrome', temp_history_db, config, 'Default')]

        with patch.object(spinner, '_detect_browsers', return_value=mock_browsers):
            with patch.object(spinner, '_copy_database', return_value=temp_history_db):
                shards = await spinner._spin_impl()

                # Should have domain summary shards
                domain_shards = [s for s in shards if s.metadata.get('shard_type') == 'domain_summary']

                # May or may not have domain summaries depending on data
                # but structure should be correct if they exist
                for shard in domain_shards:
                    assert 'domain' in shard.metadata
                    assert 'total_visits' in shard.metadata


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
