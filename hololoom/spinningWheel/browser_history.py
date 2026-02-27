"""
Browser History Spinner - Multi-Browser History Ingestion
==========================================================

Ingests browsing history from Chrome, Firefox, Edge, Safari, and Brave browsers.
Extracts URLs, titles, visit counts, timestamps, and calculates importance based
on visit frequency, recency, and domain authority.

Features:
- Auto-detects installed browsers (Chrome, Firefox, Edge, Safari, Brave)
- Reads SQLite history databases safely (copies locked files)
- Cross-platform support (Windows, macOS, Linux)
- Incremental updates via checkpointing
- Importance scoring based on visit frequency, recency, domain
- Domain-level aggregation for overview shards
- Supports multiple browser profiles

Requirements:
    None (uses standard library sqlite3)

Author: Claude Code
Date: December 2025
"""

import os
import re
import sys
import glob
import shutil
import sqlite3
import asyncio
import tempfile
import platform
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlparse
import warnings

from hololoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinResult,
    SpinnerCapabilities,
    SpinnerCheckpoint,
    ImportanceSignals,
    ImportanceScore
)
from hololoom.protocols.types import MemoryShard


# =============================================================================
# Browser Configuration
# =============================================================================

@dataclass
class BrowserConfig:
    """Configuration for a specific browser."""
    name: str
    history_paths: Dict[str, List[str]]  # Platform -> list of path patterns
    history_table: str  # Table name in SQLite DB
    url_column: str  # Column name for URL
    title_column: str  # Column name for title
    visit_count_column: str  # Column name for visit count
    last_visit_column: str  # Column name for last visit time
    timestamp_format: str  # 'webkit' (Chrome/Edge/Brave) or 'prtime' (Firefox) or 'mac_absolute' (Safari)


# Browser configurations
BROWSER_CONFIGS: Dict[str, BrowserConfig] = {
    'chrome': BrowserConfig(
        name='Chrome',
        history_paths={
            'Windows': [os.path.expandvars(r'%LOCALAPPDATA%\Google\Chrome\User Data\*\History')],
            'Darwin': [os.path.expanduser('~/Library/Application Support/Google/Chrome/*/History')],
            'Linux': [os.path.expanduser('~/.config/google-chrome/*/History')]
        },
        history_table='urls',
        url_column='url',
        title_column='title',
        visit_count_column='visit_count',
        last_visit_column='last_visit_time',
        timestamp_format='webkit'
    ),
    'firefox': BrowserConfig(
        name='Firefox',
        history_paths={
            'Windows': [os.path.expandvars(r'%APPDATA%\Mozilla\Firefox\Profiles\*\places.sqlite')],
            'Darwin': [os.path.expanduser('~/Library/Application Support/Firefox/Profiles/*/places.sqlite')],
            'Linux': [os.path.expanduser('~/.mozilla/firefox/*/places.sqlite')]
        },
        history_table='moz_places',
        url_column='url',
        title_column='title',
        visit_count_column='visit_count',
        last_visit_column='last_visit_date',
        timestamp_format='prtime'
    ),
    'edge': BrowserConfig(
        name='Edge',
        history_paths={
            'Windows': [os.path.expandvars(r'%LOCALAPPDATA%\Microsoft\Edge\User Data\*\History')],
            'Darwin': [os.path.expanduser('~/Library/Application Support/Microsoft Edge/*/History')],
            'Linux': [os.path.expanduser('~/.config/microsoft-edge/*/History')]
        },
        history_table='urls',
        url_column='url',
        title_column='title',
        visit_count_column='visit_count',
        last_visit_column='last_visit_time',
        timestamp_format='webkit'
    ),
    'safari': BrowserConfig(
        name='Safari',
        history_paths={
            'Darwin': [os.path.expanduser('~/Library/Safari/History.db')]
        },
        history_table='history_items',
        url_column='url',
        title_column='title',  # Safari stores titles differently
        visit_count_column='visit_count',
        last_visit_column='visit_time',
        timestamp_format='mac_absolute'
    ),
    'brave': BrowserConfig(
        name='Brave',
        history_paths={
            'Windows': [os.path.expandvars(r'%LOCALAPPDATA%\BraveSoftware\Brave-Browser\User Data\*\History')],
            'Darwin': [os.path.expanduser('~/Library/Application Support/BraveSoftware/Brave-Browser/*/History')],
            'Linux': [os.path.expanduser('~/.config/BraveSoftware/Brave-Browser/*/History')]
        },
        history_table='urls',
        url_column='url',
        title_column='title',
        visit_count_column='visit_count',
        last_visit_column='last_visit_time',
        timestamp_format='webkit'
    )
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HistoryEntry:
    """Single browser history entry."""
    url: str
    title: str
    visit_count: int
    last_visit: datetime
    browser: str
    profile: str = 'Default'
    domain: str = ''

    def __post_init__(self):
        """Extract domain from URL."""
        if not self.domain:
            try:
                parsed = urlparse(self.url)
                self.domain = parsed.netloc or parsed.path.split('/')[0]
            except Exception:
                self.domain = 'unknown'


@dataclass
class DomainStats:
    """Aggregated stats for a domain."""
    domain: str
    total_visits: int = 0
    unique_pages: int = 0
    last_visit: Optional[datetime] = None
    first_seen: Optional[datetime] = None
    titles: List[str] = field(default_factory=list)
    browsers: List[str] = field(default_factory=list)


# =============================================================================
# Timestamp Conversion
# =============================================================================

def convert_webkit_timestamp(timestamp: int) -> datetime:
    """
    Convert WebKit timestamp to datetime.

    Chrome/Edge/Brave use microseconds since Jan 1, 1601 (WebKit epoch).
    """
    if not timestamp or timestamp <= 0:
        return datetime.now()

    # WebKit epoch: Jan 1, 1601
    # Unix epoch: Jan 1, 1970
    # Difference: 11644473600 seconds
    WEBKIT_EPOCH_DIFF = 11644473600

    try:
        # Convert microseconds to seconds
        unix_seconds = (timestamp / 1_000_000) - WEBKIT_EPOCH_DIFF
        return datetime.fromtimestamp(max(0, unix_seconds))
    except (ValueError, OSError):
        return datetime.now()


def convert_prtime_timestamp(timestamp: int) -> datetime:
    """
    Convert Mozilla PRTime to datetime.

    Firefox uses microseconds since Jan 1, 1970 (Unix epoch).
    """
    if not timestamp or timestamp <= 0:
        return datetime.now()

    try:
        # Convert microseconds to seconds
        unix_seconds = timestamp / 1_000_000
        return datetime.fromtimestamp(max(0, unix_seconds))
    except (ValueError, OSError):
        return datetime.now()


def convert_mac_absolute_timestamp(timestamp: float) -> datetime:
    """
    Convert Mac Absolute Time to datetime.

    Safari uses seconds since Jan 1, 2001 (Mac epoch).
    """
    if not timestamp or timestamp <= 0:
        return datetime.now()

    # Mac epoch: Jan 1, 2001
    # Unix epoch: Jan 1, 1970
    # Difference: 978307200 seconds
    MAC_EPOCH_DIFF = 978307200

    try:
        unix_seconds = timestamp + MAC_EPOCH_DIFF
        return datetime.fromtimestamp(max(0, unix_seconds))
    except (ValueError, OSError):
        return datetime.now()


def convert_timestamp(timestamp: Any, format_type: str) -> datetime:
    """Convert timestamp based on format type."""
    if timestamp is None:
        return datetime.now()

    converters = {
        'webkit': convert_webkit_timestamp,
        'prtime': convert_prtime_timestamp,
        'mac_absolute': convert_mac_absolute_timestamp
    }

    converter = converters.get(format_type, lambda x: datetime.now())
    return converter(timestamp)


# =============================================================================
# Browser History Spinner
# =============================================================================

class BrowserHistorySpinner(BaseSpinner):
    """
    Ingest browser history from multiple browsers.

    Features:
    - Auto-detects installed browsers
    - Reads history safely (copies locked database files)
    - Cross-platform support
    - Importance scoring based on visit frequency, recency, domain
    - Domain-level aggregation
    - Incremental updates via checkpointing
    """

    def __init__(
        self,
        importance_threshold: float = 0.3,
        browsers: Optional[List[str]] = None,
        days_back: int = 30,
        min_visits: int = 1,
        include_domains_only: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        aggregate_by_domain: bool = True,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize BrowserHistorySpinner.

        Args:
            importance_threshold: Minimum importance score to include
            browsers: List of browsers to read from (None = auto-detect all)
            days_back: How many days of history to read (default: 30)
            min_visits: Minimum visit count to include (default: 1)
            include_domains_only: If set, only include these domains
            exclude_domains: Domains to exclude (e.g., ['google.com', 'facebook.com'])
            aggregate_by_domain: Create domain-level summary shards (default: True)
            checkpoint_dir: Directory for incremental checkpoints
        """
        super().__init__(
            name='browser_history',
            importance_threshold=importance_threshold,
            checkpoint_dir=checkpoint_dir
        )

        self.browsers = browsers
        self.days_back = days_back
        self.min_visits = min_visits
        self.include_domains_only = set(include_domains_only) if include_domains_only else None
        self.exclude_domains = set(exclude_domains or [
            'google.com', 'www.google.com',
            'youtube.com', 'www.youtube.com',
            'facebook.com', 'www.facebook.com',
            'twitter.com', 'x.com',
            'instagram.com', 'www.instagram.com',
            'linkedin.com', 'www.linkedin.com',
            'reddit.com', 'www.reddit.com',
            'amazon.com', 'www.amazon.com',
            'localhost', '127.0.0.1'
        ])
        self.aggregate_by_domain = aggregate_by_domain

    def get_capabilities(self) -> SpinnerCapabilities:
        """Get spinner capabilities."""
        return SpinnerCapabilities(
            basic_processing=True,
            streaming=False,
            incremental=True,  # Supports checkpointing
            batch_processing=True,
            importance_scoring=True,
            entity_extraction=False,
            motif_extraction=False,
            supported_formats=['chrome', 'firefox', 'edge', 'safari', 'brave']
        )

    def is_available(self) -> bool:
        """Check if any browser history is accessible."""
        return len(self._detect_browsers()) > 0

    def _detect_browsers(self) -> List[Tuple[str, str, BrowserConfig]]:
        """
        Detect installed browsers and their history files.

        Returns:
            List of (browser_name, history_path, config) tuples
        """
        current_platform = platform.system()
        detected = []

        browsers_to_check = self.browsers or list(BROWSER_CONFIGS.keys())

        for browser_name in browsers_to_check:
            if browser_name not in BROWSER_CONFIGS:
                continue

            config = BROWSER_CONFIGS[browser_name]

            # Get paths for current platform
            path_patterns = config.history_paths.get(current_platform, [])

            for pattern in path_patterns:
                # Expand glob patterns
                matches = glob.glob(pattern)
                for match in matches:
                    if os.path.isfile(match):
                        # Extract profile name from path
                        profile = self._extract_profile_name(match)
                        detected.append((browser_name, match, config, profile))

        return detected

    def _extract_profile_name(self, path: str) -> str:
        """Extract profile name from history file path."""
        parts = Path(path).parts

        # Look for common profile folder names
        for part in parts:
            if part.startswith('Profile ') or part == 'Default':
                return part
            if '.default' in part or 'default-release' in part:
                return part

        return 'Default'

    def _copy_database(self, source_path: str) -> str:
        """
        Copy database to temp file (browsers lock their databases).

        Returns:
            Path to temporary copy
        """
        temp_dir = tempfile.gettempdir()
        temp_name = f"hololoom_history_{os.getpid()}_{hash(source_path) % 10000}.db"
        temp_path = os.path.join(temp_dir, temp_name)

        try:
            shutil.copy2(source_path, temp_path)
            return temp_path
        except (PermissionError, OSError) as e:
            warnings.warn(f"Could not copy database {source_path}: {e}")
            return None

    def _read_history(
        self,
        db_path: str,
        config: BrowserConfig,
        browser_name: str,
        profile: str,
        since_timestamp: Optional[datetime] = None
    ) -> List[HistoryEntry]:
        """
        Read history entries from a browser database.

        Args:
            db_path: Path to the (copied) database file
            config: Browser configuration
            browser_name: Browser name
            profile: Profile name
            since_timestamp: Only read entries after this time

        Returns:
            List of HistoryEntry objects
        """
        entries = []

        try:
            conn = sqlite3.connect(db_path, timeout=10)
            cursor = conn.cursor()

            # Calculate cutoff time
            cutoff = datetime.now() - timedelta(days=self.days_back)
            if since_timestamp and since_timestamp > cutoff:
                cutoff = since_timestamp

            # Build query based on browser
            query = f"""
                SELECT {config.url_column},
                       {config.title_column},
                       {config.visit_count_column},
                       {config.last_visit_column}
                FROM {config.history_table}
                WHERE {config.visit_count_column} >= ?
            """

            params = [self.min_visits]

            cursor.execute(query, params)
            rows = cursor.fetchall()

            for row in rows:
                url, title, visit_count, last_visit = row

                # Skip empty URLs
                if not url:
                    continue

                # Convert timestamp
                last_visit_dt = convert_timestamp(last_visit, config.timestamp_format)

                # Skip old entries
                if last_visit_dt < cutoff:
                    continue

                # Create entry
                entry = HistoryEntry(
                    url=url,
                    title=title or url,
                    visit_count=visit_count or 1,
                    last_visit=last_visit_dt,
                    browser=browser_name,
                    profile=profile
                )

                # Apply domain filters
                if self.include_domains_only:
                    if entry.domain not in self.include_domains_only:
                        continue

                if entry.domain in self.exclude_domains:
                    continue

                # Skip internal browser pages
                if any(url.startswith(prefix) for prefix in [
                    'chrome://', 'chrome-extension://',
                    'edge://', 'about:',
                    'file://', 'moz-extension://'
                ]):
                    continue

                entries.append(entry)

            conn.close()

        except sqlite3.Error as e:
            warnings.warn(f"SQLite error reading {browser_name}: {e}")
        except Exception as e:
            warnings.warn(f"Error reading {browser_name} history: {e}")

        return entries

    def score_importance(self, entry: HistoryEntry) -> ImportanceScore:
        """
        Score importance of a history entry.

        Scoring factors:
        - Visit count (more visits = more important)
        - Recency (recent visits = more important)
        - Domain authority (educational, documentation sites)
        - Title quality (has meaningful title)
        """
        signals = ImportanceSignals()

        # Visit count score (logarithmic, capped at 100 visits)
        visit_score = min(1.0, (entry.visit_count / 10) ** 0.5)
        signals.length_score = visit_score  # Using length_score for visits

        # Recency score (exponential decay over days_back)
        days_old = (datetime.now() - entry.last_visit).days
        recency_score = max(0.0, 1.0 - (days_old / self.days_back))
        signals.recency_score = recency_score

        # Domain authority score
        authority_domains = {
            'github.com': 0.9,
            'stackoverflow.com': 0.85,
            'docs.': 0.9,  # documentation sites
            'wiki': 0.75,
            '.edu': 0.85,
            '.gov': 0.8,
            'medium.com': 0.6,
            'dev.to': 0.7,
            'arxiv.org': 0.9,
            'readthedocs.io': 0.85,
        }

        authority = 0.5  # Default
        domain_lower = entry.domain.lower()
        for pattern, score in authority_domains.items():
            if pattern in domain_lower:
                authority = max(authority, score)
                break
        signals.authority_score = authority

        # Title quality score
        if entry.title and entry.title != entry.url:
            # Has a real title
            title_len = len(entry.title)
            if title_len > 20:
                signals.technical_score = 0.7
            elif title_len > 5:
                signals.technical_score = 0.5
            else:
                signals.technical_score = 0.3
        else:
            signals.technical_score = 0.2

        # Compute total
        total = signals.compute_total()

        return ImportanceScore(
            score=total,
            signals=signals,
            reason=f"visits={entry.visit_count}, recency={recency_score:.2f}, authority={authority:.2f}"
        )

    async def _spin_impl(
        self,
        source: Any = None,
        **kwargs
    ) -> List[MemoryShard]:
        """
        Read browser history and create memory shards.

        Args:
            source: Not used (reads from detected browsers)
            **kwargs: Additional arguments

        Returns:
            List of MemoryShard objects
        """
        # Detect available browsers
        browsers = self._detect_browsers()

        if not browsers:
            warnings.warn("No browser history found")
            return []

        # Read history from all browsers
        all_entries: List[HistoryEntry] = []

        for browser_name, history_path, config, profile in browsers:
            # Copy database to avoid locking issues
            temp_db = self._copy_database(history_path)
            if not temp_db:
                continue

            try:
                entries = self._read_history(
                    temp_db, config, browser_name, profile
                )
                all_entries.extend(entries)
            finally:
                # Clean up temp file
                try:
                    os.remove(temp_db)
                except OSError:
                    pass

        # Sort by visit count (descending)
        all_entries.sort(key=lambda e: e.visit_count, reverse=True)

        # Create shards
        shards = []

        # Individual page shards (top entries only to avoid overwhelming)
        for entry in all_entries[:500]:  # Limit to top 500 pages
            importance = self.score_importance(entry)

            shard = MemoryShard(
                id=f"browser_history_{hash(entry.url) % 1000000}",
                text=f"{entry.title}\n\nURL: {entry.url}\n\nVisited {entry.visit_count} times. Last visit: {entry.last_visit.strftime('%Y-%m-%d %H:%M')}",
                episode=f"browser_history_{entry.browser}",
                entities=[entry.domain],
                motifs=['browsing', entry.browser.lower()],
                metadata={
                    'url': entry.url,
                    'title': entry.title,
                    'domain': entry.domain,
                    'visit_count': entry.visit_count,
                    'last_visit': entry.last_visit.isoformat(),
                    'browser': entry.browser,
                    'profile': entry.profile,
                    'importance_score': importance.score,
                    'importance_reason': importance.reason,
                    'shard_type': 'page_history'
                }
            )
            shards.append(shard)

        # Domain aggregation shards
        if self.aggregate_by_domain:
            domain_stats = self._aggregate_by_domain(all_entries)

            for domain, stats in sorted(
                domain_stats.items(),
                key=lambda x: x[1].total_visits,
                reverse=True
            )[:100]:  # Top 100 domains

                # Calculate domain importance
                domain_importance = min(1.0, (stats.total_visits / 50) ** 0.5) * 0.7 + \
                                   min(1.0, stats.unique_pages / 20) * 0.3

                sample_titles = stats.titles[:5]
                title_list = '\n- '.join(sample_titles) if sample_titles else 'No titles'

                shard = MemoryShard(
                    id=f"browser_domain_{hash(domain) % 1000000}",
                    text=f"Domain: {domain}\n\nTotal visits: {stats.total_visits}\nUnique pages: {stats.unique_pages}\nBrowsers: {', '.join(set(stats.browsers))}\n\nSample pages:\n- {title_list}",
                    episode='browser_history_domains',
                    entities=[domain],
                    motifs=['browsing', 'domain_summary'],
                    metadata={
                        'domain': domain,
                        'total_visits': stats.total_visits,
                        'unique_pages': stats.unique_pages,
                        'last_visit': stats.last_visit.isoformat() if stats.last_visit else None,
                        'browsers': list(set(stats.browsers)),
                        'importance_score': domain_importance,
                        'shard_type': 'domain_summary'
                    }
                )
                shards.append(shard)

        return shards

    def _aggregate_by_domain(
        self,
        entries: List[HistoryEntry]
    ) -> Dict[str, DomainStats]:
        """Aggregate history entries by domain."""
        domain_stats: Dict[str, DomainStats] = {}

        for entry in entries:
            if entry.domain not in domain_stats:
                domain_stats[entry.domain] = DomainStats(domain=entry.domain)

            stats = domain_stats[entry.domain]
            stats.total_visits += entry.visit_count
            stats.unique_pages += 1
            stats.browsers.append(entry.browser)

            if entry.title and entry.title != entry.url:
                stats.titles.append(entry.title)

            if stats.last_visit is None or entry.last_visit > stats.last_visit:
                stats.last_visit = entry.last_visit

            if stats.first_seen is None or entry.last_visit < stats.first_seen:
                stats.first_seen = entry.last_visit

        return domain_stats

    async def spin_incremental(
        self,
        source: Any = None,
        checkpoint: Optional[SpinnerCheckpoint] = None,
        **kwargs
    ) -> SpinResult:
        """
        Read only new history entries since last checkpoint.

        Args:
            source: Not used
            checkpoint: Previous checkpoint (None = full read)
            **kwargs: Additional arguments

        Returns:
            SpinResult with only new entries
        """
        since_timestamp = None

        if checkpoint:
            # Resume from checkpoint
            since_timestamp = datetime.fromtimestamp(
                checkpoint.last_processed_timestamp or 0
            )

        # Temporarily adjust the cutoff
        original_days_back = self.days_back

        if since_timestamp:
            # Only read since last checkpoint
            days_since = (datetime.now() - since_timestamp).days
            self.days_back = max(1, days_since + 1)

        try:
            result = await self.spin(source, **kwargs)

            # Create new checkpoint
            if result.success and result.shards:
                new_checkpoint = SpinnerCheckpoint(
                    spinner_name=self.name,
                    source_id='browser_history',
                    last_processed_timestamp=datetime.now().timestamp(),
                    items_processed=(checkpoint.items_processed if checkpoint else 0) + result.shard_count
                )
                self.save_checkpoint(new_checkpoint)

            return result

        finally:
            self.days_back = original_days_back


# =============================================================================
# Convenience Functions
# =============================================================================

async def read_browser_history(
    browsers: Optional[List[str]] = None,
    days_back: int = 30,
    min_visits: int = 2,
    exclude_common: bool = True
) -> SpinResult:
    """
    Convenience function to read browser history.

    Args:
        browsers: List of browsers (None = auto-detect all)
        days_back: Days of history to read
        min_visits: Minimum visit count
        exclude_common: Exclude common social/search domains

    Returns:
        SpinResult with history shards
    """
    exclude_domains = None if not exclude_common else [
        'google.com', 'www.google.com',
        'youtube.com', 'www.youtube.com',
        'facebook.com', 'www.facebook.com',
        'twitter.com', 'x.com',
        'instagram.com', 'www.instagram.com',
        'linkedin.com', 'www.linkedin.com'
    ]

    spinner = BrowserHistorySpinner(
        importance_threshold=0.2,
        browsers=browsers,
        days_back=days_back,
        min_visits=min_visits,
        exclude_domains=exclude_domains
    )

    return await spinner.spin(None)


def get_available_browsers() -> List[str]:
    """
    Get list of browsers with accessible history.

    Returns:
        List of browser names (e.g., ['chrome', 'firefox'])
    """
    spinner = BrowserHistorySpinner(importance_threshold=0.0)
    browsers = spinner._detect_browsers()
    return list(set(b[0] for b in browsers))


# Alias for consistency
spin_browser_history = read_browser_history
