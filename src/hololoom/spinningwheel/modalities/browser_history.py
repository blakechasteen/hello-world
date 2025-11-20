"""
Browser History Reader - Extract browsing history from Chrome, Firefox, Edge.

Reads browser history databases and provides structured access to:
- URLs visited
- Page titles
- Visit timestamps
- Visit duration
- Visit count

Supports:
- Chrome/Chromium
- Firefox
- Microsoft Edge
- Brave
"""

import sqlite3
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class BrowserVisit:
    """A single browser history visit."""
    url: str
    title: str
    timestamp: datetime
    duration: int  # seconds
    visit_count: int


class BrowserHistoryReader:
    """Read browser history from local databases."""

    # Browser database paths (Windows)
    CHROME_PATHS = [
        Path.home() / "AppData/Local/Google/Chrome/User Data/Default/History",
        Path.home() / "AppData/Local/Google/Chrome/User Data/Profile 1/History",
    ]

    EDGE_PATHS = [
        Path.home() / "AppData/Local/Microsoft/Edge/User Data/Default/History",
    ]

    BRAVE_PATHS = [
        Path.home() / "AppData/Local/BraveSoftware/Brave-Browser/User Data/Default/History",
    ]

    FIREFOX_PATHS = [
        Path.home() / "AppData/Roaming/Mozilla/Firefox/Profiles",  # Directory with profiles
    ]

    @staticmethod
    def _chrome_timestamp_to_datetime(chrome_timestamp: int) -> datetime:
        """
        Convert Chrome timestamp to datetime.

        Chrome uses microseconds since 1601-01-01 (Windows epoch).
        """
        # Chrome epoch: January 1, 1601
        epoch = datetime(1601, 1, 1)
        return epoch + timedelta(microseconds=chrome_timestamp)

    @staticmethod
    def _copy_db(db_path: Path) -> Path:
        """
        Copy database to temp location to avoid locking issues.

        Browsers lock their history databases while running.
        """
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        temp_dir = tempfile.gettempdir()
        temp_path = Path(temp_dir) / f"hololoom_history_{db_path.name}"

        shutil.copy2(db_path, temp_path)
        return temp_path

    @classmethod
    def read_chrome_history(
        cls,
        days_back: int = 7,
        min_duration: int = 0,
        profile_path: Optional[Path] = None
    ) -> List[BrowserVisit]:
        """
        Read Chrome browser history.

        Args:
            days_back: How many days of history to retrieve
            min_duration: Minimum visit duration (seconds) to include
            profile_path: Optional custom profile path

        Returns:
            List of BrowserVisit objects
        """
        # Find history database
        db_path = profile_path
        if not db_path:
            for path in cls.CHROME_PATHS:
                if path.exists():
                    db_path = path
                    break

        if not db_path or not db_path.exists():
            logger.warning("Chrome history database not found")
            return []

        try:
            # Copy to avoid locking
            temp_db = cls._copy_db(db_path)

            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()

            # Calculate cutoff timestamp
            cutoff = datetime.now() - timedelta(days=days_back)
            # Chrome uses microseconds since 1601
            chrome_epoch = datetime(1601, 1, 1)
            cutoff_timestamp = int((cutoff - chrome_epoch).total_seconds() * 1000000)

            # Query history
            cursor.execute("""
                SELECT
                    urls.url,
                    urls.title,
                    visits.visit_time,
                    visits.visit_duration,
                    urls.visit_count
                FROM urls
                INNER JOIN visits ON urls.id = visits.url
                WHERE visits.visit_time > ?
                    AND visits.visit_duration >= ?
                ORDER BY visits.visit_time DESC
            """, (cutoff_timestamp, min_duration * 1000000))  # Duration in microseconds

            results = []
            for row in cursor.fetchall():
                url, title, visit_time, visit_duration, visit_count = row

                # Convert Chrome timestamp
                timestamp = cls._chrome_timestamp_to_datetime(visit_time)

                # Convert duration to seconds
                duration_seconds = visit_duration // 1000000 if visit_duration else 0

                results.append(BrowserVisit(
                    url=url,
                    title=title or "",
                    timestamp=timestamp,
                    duration=duration_seconds,
                    visit_count=visit_count
                ))

            conn.close()
            temp_db.unlink()  # Clean up temp file

            logger.info(f"Read {len(results)} Chrome history entries")
            return results

        except Exception as e:
            logger.error(f"Error reading Chrome history: {e}")
            return []

    @classmethod
    def read_edge_history(
        cls,
        days_back: int = 7,
        min_duration: int = 0
    ) -> List[BrowserVisit]:
        """
        Read Microsoft Edge history.

        Edge uses same format as Chrome (Chromium-based).
        """
        for path in cls.EDGE_PATHS:
            if path.exists():
                return cls.read_chrome_history(
                    days_back=days_back,
                    min_duration=min_duration,
                    profile_path=path
                )

        logger.warning("Edge history database not found")
        return []

    @classmethod
    def read_brave_history(
        cls,
        days_back: int = 7,
        min_duration: int = 0
    ) -> List[BrowserVisit]:
        """
        Read Brave browser history.

        Brave uses same format as Chrome (Chromium-based).
        """
        for path in cls.BRAVE_PATHS:
            if path.exists():
                return cls.read_chrome_history(
                    days_back=days_back,
                    min_duration=min_duration,
                    profile_path=path
                )

        logger.warning("Brave history database not found")
        return []

    @classmethod
    def read_firefox_history(
        cls,
        days_back: int = 7,
        min_duration: int = 0
    ) -> List[BrowserVisit]:
        """
        Read Firefox browser history.

        Firefox uses a different database schema than Chrome.
        """
        # Find Firefox profiles
        profiles_dir = cls.FIREFOX_PATHS[0]
        if not profiles_dir.exists():
            logger.warning("Firefox profiles directory not found")
            return []

        # Find default profile (ends with .default-release)
        profile_dirs = [p for p in profiles_dir.iterdir() if p.is_dir() and 'default' in p.name.lower()]

        if not profile_dirs:
            logger.warning("No Firefox profiles found")
            return []

        # Use first profile found
        db_path = profile_dirs[0] / "places.sqlite"
        if not db_path.exists():
            logger.warning(f"Firefox history database not found: {db_path}")
            return []

        try:
            temp_db = cls._copy_db(db_path)

            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()

            # Firefox uses microseconds since Unix epoch
            cutoff = datetime.now() - timedelta(days=days_back)
            cutoff_timestamp = int(cutoff.timestamp() * 1000000)

            # Query Firefox history
            cursor.execute("""
                SELECT
                    moz_places.url,
                    moz_places.title,
                    moz_historyvisits.visit_date,
                    moz_places.visit_count
                FROM moz_places
                INNER JOIN moz_historyvisits ON moz_places.id = moz_historyvisits.place_id
                WHERE moz_historyvisits.visit_date > ?
                ORDER BY moz_historyvisits.visit_date DESC
            """, (cutoff_timestamp,))

            results = []
            for row in cursor.fetchall():
                url, title, visit_date, visit_count = row

                # Convert Firefox timestamp (microseconds since Unix epoch)
                timestamp = datetime.fromtimestamp(visit_date / 1000000)

                # Firefox doesn't track duration, estimate based on gaps
                duration = 0  # Could estimate from visit gaps

                results.append(BrowserVisit(
                    url=url,
                    title=title or "",
                    timestamp=timestamp,
                    duration=duration,
                    visit_count=visit_count
                ))

            conn.close()
            temp_db.unlink()

            logger.info(f"Read {len(results)} Firefox history entries")
            return results

        except Exception as e:
            logger.error(f"Error reading Firefox history: {e}")
            return []

    @classmethod
    def read_all_browsers(
        cls,
        days_back: int = 7,
        min_duration: int = 0
    ) -> Dict[str, List[BrowserVisit]]:
        """
        Read history from all available browsers.

        Returns:
            Dict mapping browser name to list of visits
        """
        results = {}

        # Try each browser
        chrome = cls.read_chrome_history(days_back, min_duration)
        if chrome:
            results['chrome'] = chrome

        edge = cls.read_edge_history(days_back, min_duration)
        if edge:
            results['edge'] = edge

        brave = cls.read_brave_history(days_back, min_duration)
        if brave:
            results['brave'] = brave

        firefox = cls.read_firefox_history(days_back, min_duration)
        if firefox:
            results['firefox'] = firefox

        return results

    @classmethod
    def filter_meaningful_visits(
        cls,
        visits: List[BrowserVisit],
        min_duration: int = 30,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[BrowserVisit]:
        """
        Filter to only meaningful visits.

        Args:
            visits: List of browser visits
            min_duration: Minimum time spent on page (seconds)
            exclude_patterns: URL patterns to exclude (e.g., ['google.com/search'])

        Returns:
            Filtered list of visits
        """
        exclude_patterns = exclude_patterns or [
            'google.com/search',
            'bing.com/search',
            'duckduckgo.com',
            'facebook.com',
            'twitter.com',
            'reddit.com/r/all',
            'youtube.com/feed',
        ]

        filtered = []
        for visit in visits:
            # Check duration
            if visit.duration < min_duration:
                continue

            # Check exclude patterns
            if any(pattern in visit.url for pattern in exclude_patterns):
                continue

            filtered.append(visit)

        return filtered


# Convenience function
def get_recent_history(
    days_back: int = 7,
    min_duration: int = 30,
    browser: str = 'chrome'
) -> List[BrowserVisit]:
    """
    Get recent meaningful browser history.

    Args:
        days_back: How many days to look back
        min_duration: Minimum page visit time (seconds)
        browser: 'chrome', 'edge', 'brave', 'firefox', or 'all'

    Returns:
        List of BrowserVisit objects

    Example:
        # Get last week of Chrome history (pages visited for 30+ seconds)
        visits = get_recent_history(days_back=7, min_duration=30)

        for visit in visits:
            print(f"{visit.title} - {visit.url}")
    """
    reader = BrowserHistoryReader()

    if browser == 'all':
        all_visits = reader.read_all_browsers(days_back, min_duration)
        # Combine and deduplicate
        combined = []
        seen_urls = set()
        for visits in all_visits.values():
            for visit in visits:
                if visit.url not in seen_urls:
                    combined.append(visit)
                    seen_urls.add(visit.url)
        return combined

    elif browser == 'chrome':
        return reader.read_chrome_history(days_back, min_duration)
    elif browser == 'edge':
        return reader.read_edge_history(days_back, min_duration)
    elif browser == 'brave':
        return reader.read_brave_history(days_back, min_duration)
    elif browser == 'firefox':
        return reader.read_firefox_history(days_back, min_duration)
    else:
        raise ValueError(f"Unknown browser: {browser}")
