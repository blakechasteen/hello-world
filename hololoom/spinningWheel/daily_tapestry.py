"""
Daily Tapestry — YouTube Playlist Briefing
===========================================

Scans YouTube playlists, ingests new video transcripts, and weaves
a structured briefing (tapestry) from the results. The tapestry
highlights cross-playlist resonance (videos you cared enough to
bookmark from multiple angles), clusters by topic, summarizes via
LLM, and tracks week-over-week trends.

Four interfaces, single core:
  1. CLI:  python -m hololoom.spinningWheel.daily_tapestry
  2. Cron: schedule the CLI as a daily job
  3. IPC:  drop a JSON file in Femtoclaw's IPC dir for Matrix delivery
  4. Standing Intent: wire into Layer 5 heartbeat when implemented

Output routes:
  - stdout (default)
  - vault federation proposal (--vault)
  - Matrix room via Femtoclaw IPC (--matrix ROOM_JID)
  - JSON (--json, for downstream consumers)

Requirements:
    pip install yt-dlp youtube-transcript-api

Optional (for summarization):
    Ollama running with qwen3.5:9b

Optional (for clustering):
    pip install hololoom[ml]  (numpy, scikit-learn)

Author: Claude Code
Date: March 2026
"""

import argparse
import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hololoom.spinningWheel.playlist_spinner import PlaylistSpinner, ResonanceIndex
from hololoom.spinningWheel.protocol import SpinResult

logger = logging.getLogger(__name__)


# =============================================================================
# Optional dependencies (graceful degradation)
# =============================================================================

try:
    from hololoom.apps.server.kit.llm import OllamaBackend, extract_content
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

try:
    from hololoom.core.memory.clustering import cluster as cluster_texts, ClusteringOutput
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CHECKPOINT_DIR = Path(
    os.environ.get(
        'TAPESTRY_CHECKPOINT_DIR',
        Path.home() / '.hololoom' / 'tapestry',
    )
)

DEFAULT_PLAYLIST_FILE = Path(
    os.environ.get(
        'TAPESTRY_PLAYLISTS',
        Path.home() / '.hololoom' / 'tapestry' / 'playlists.txt',
    )
)

OLLAMA_MODEL = os.environ.get('TAPESTRY_MODEL', 'qwen3.5:9b')

# Femtoclaw IPC (WSL path by default)
FEMTOCLAW_IPC_DIR = Path(
    os.environ.get(
        'FEMTOCLAW_IPC_DIR',
        '/home/blake/nanoclaw/data/ipc',
    )
)


def load_playlist_urls(path: Path | None = None) -> list[str]:
    """Load playlist URLs from a text file (one URL per line)."""
    path = path or DEFAULT_PLAYLIST_FILE
    if not path.exists():
        logger.warning("No playlist file at %s", path)
        return []

    urls = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                urls.append(line)

    return urls


# =============================================================================
# Tapestry Data Structures
# =============================================================================

@dataclass
class TapestryEntry:
    """One video in the briefing."""
    video_id: str
    title: str
    channel: str
    url: str
    resonance_count: int
    importance: float
    summary: str
    playlists: list[str] = field(default_factory=list)
    shard_type: str = 'transcript'


@dataclass
class TopicCluster:
    """A group of related videos."""
    topic: str
    entries: list[TapestryEntry] = field(default_factory=list)

    @property
    def total_resonance(self) -> int:
        return sum(e.resonance_count for e in self.entries)


@dataclass
class Trend:
    """A detected trend across runs."""
    topic: str
    current_count: int
    previous_count: int
    direction: str  # "up", "down", "new", "stable"

    def describe(self) -> str:
        if self.direction == 'new':
            return f"{self.topic}: {self.current_count} videos (new topic)"
        arrow = {'up': '^', 'down': 'v', 'stable': '='}[self.direction]
        return f"{self.topic}: {self.current_count} videos ({arrow} from {self.previous_count})"


@dataclass
class Tapestry:
    """The complete daily briefing."""
    date: str
    total_videos_scanned: int
    new_videos_processed: int
    clusters: list[TopicCluster] = field(default_factory=list)
    high_resonance: list[TapestryEntry] = field(default_factory=list)
    trends: list[Trend] = field(default_factory=list)
    skipped_count: int = 0

    def to_markdown(self) -> str:
        """Format as readable markdown briefing."""
        lines = [
            f"## Daily Tapestry -- {self.date}",
            "",
        ]

        # Trends section (if any)
        if self.trends:
            lines.append("### Trends")
            for trend in self.trends:
                lines.append(f"- {trend.describe()}")
            lines.append("")

        # High resonance section
        if self.high_resonance:
            lines.append("### High Resonance (multi-playlist)")
            for entry in self.high_resonance:
                stars = '*' * entry.resonance_count
                playlist_list = ', '.join(entry.playlists[:5])
                lines.append(f"**{entry.title}** by {entry.channel} -- {stars}")
                lines.append(f"  {entry.url}")
                lines.append(f"  Playlists: {playlist_list}")
                lines.append(f"  {entry.summary}")
                lines.append("")

        # Clustered sections
        if self.clusters:
            lines.append("### Topics")
            for cluster in self.clusters:
                lines.append(f"#### {cluster.topic} ({len(cluster.entries)} videos)")
                for entry in cluster.entries:
                    if entry.resonance_count >= 2:
                        continue  # Already shown in high resonance
                    lines.append(f"- **{entry.title}** by {entry.channel}")
                    lines.append(f"  {entry.summary}")
                    lines.append("")
        else:
            # Flat list fallback (no clustering)
            regular = [e for e in self._all_entries() if e.resonance_count <= 1]
            if regular:
                lines.append("### New Additions")
                for entry in regular:
                    lines.append(f"- **{entry.title}** by {entry.channel}")
                    lines.append(f"  {entry.url}")
                    lines.append(f"  {entry.summary}")
                    lines.append("")

        # Stats
        lines.append("### Stats")
        lines.append(f"- Videos scanned: {self.total_videos_scanned}")
        lines.append(f"- New this run: {self.new_videos_processed}")
        if self.skipped_count:
            lines.append(f"- Skipped (below threshold): {self.skipped_count}")

        return '\n'.join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        return {
            'date': self.date,
            'total_videos_scanned': self.total_videos_scanned,
            'new_videos_processed': self.new_videos_processed,
            'skipped_count': self.skipped_count,
            'trends': [
                {'topic': t.topic, 'current': t.current_count,
                 'previous': t.previous_count, 'direction': t.direction}
                for t in self.trends
            ],
            'high_resonance': [
                {
                    'video_id': e.video_id, 'title': e.title,
                    'channel': e.channel, 'url': e.url,
                    'resonance_count': e.resonance_count,
                    'importance': e.importance,
                    'playlists': e.playlists, 'summary': e.summary,
                }
                for e in self.high_resonance
            ],
            'clusters': [
                {
                    'topic': c.topic,
                    'entries': [
                        {'video_id': e.video_id, 'title': e.title,
                         'channel': e.channel, 'summary': e.summary}
                        for e in c.entries
                    ],
                }
                for c in self.clusters
            ],
        }

    def _all_entries(self) -> list[TapestryEntry]:
        entries = list(self.high_resonance)
        for c in self.clusters:
            entries.extend(c.entries)
        return entries


# =============================================================================
# Phase 1: LLM Summarization
# =============================================================================

async def summarize_transcript(text: str, title: str) -> str:
    """
    Summarize a video transcript to bullet points via Ollama.

    Fallback: first 200 chars if LLM unavailable.
    """
    if not LLM_AVAILABLE:
        logger.debug("LLM unavailable, using text truncation for '%s'", title)
        return text[:200] + ('...' if len(text) > 200 else '')

    try:
        backend = OllamaBackend(
            model=OLLAMA_MODEL,
            temperature=0.4,
            timeout=60,
            max_tokens=512,
        )
        response = await backend.call([
            {
                "role": "system",
                "content": (
                    "Summarize video transcripts concisely. "
                    "Output 5-8 bullet points. No preamble."
                ),
            },
            {
                "role": "user",
                "content": f"Video: {title}\n\nTranscript:\n{text[:6000]}\n\nBullet summary:",
            },
        ])
        summary = extract_content(response)
        if summary:
            return summary
    except Exception as e:
        logger.warning("Summarization failed for '%s': %s", title, e)

    # Fallback
    return text[:200] + ('...' if len(text) > 200 else '')


# =============================================================================
# Phase 2: Topic Clustering
# =============================================================================

def cluster_entries(entries: list[TapestryEntry]) -> list[TopicCluster]:
    """
    Group entries by topic using HoloLoom clustering.

    Fallback: single "Uncategorized" cluster if deps unavailable.
    """
    if not CLUSTERING_AVAILABLE or len(entries) < 3:
        return []  # Not enough to cluster meaningfully

    try:
        texts = [e.summary for e in entries]
        output = cluster_texts(
            texts,
            k=None,
            scale=192,
            label_clusters=True,
            use_thompson=True,
        )

        clusters = []
        for result in output:
            cluster_entries_list = [entries[i] for i in result.indices]
            clusters.append(TopicCluster(
                topic=result.label,
                entries=cluster_entries_list,
            ))

        return clusters

    except Exception as e:
        logger.warning("Clustering failed: %s", e)
        return []


# =============================================================================
# Phase 3: Trend Detection
# =============================================================================

TREND_LOG_FILE = 'trend_log.json'
TREND_WINDOW_DAYS = 7
TREND_MAX_AGE_DAYS = 30


@dataclass
class TrendLog:
    """Persistent log of topic counts per run."""
    entries: list[dict[str, Any]] = field(default_factory=list)

    def append_run(self, date: str, topic_counts: dict[str, int]):
        """Record this run's topic counts."""
        self.entries.append({
            'date': date,
            'topics': topic_counts,
        })
        self._prune()

    def _prune(self):
        """Remove entries older than max age."""
        cutoff = datetime.now(timezone.utc).timestamp() - (TREND_MAX_AGE_DAYS * 86400)
        self.entries = [
            e for e in self.entries
            if _parse_date(e['date']).timestamp() > cutoff
        ]

    def previous_counts(self, window_days: int = TREND_WINDOW_DAYS) -> dict[str, int]:
        """Aggregate topic counts from previous runs within the window."""
        cutoff = datetime.now(timezone.utc).timestamp() - (window_days * 86400)
        counts: dict[str, int] = defaultdict(int)
        for entry in self.entries[:-1]:  # Exclude current run
            if _parse_date(entry['date']).timestamp() > cutoff:
                for topic, count in entry.get('topics', {}).items():
                    counts[topic] += count
        return dict(counts)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'entries': self.entries}, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'TrendLog':
        if not path.exists():
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(entries=data.get('entries', []))
        except Exception:
            return cls()


def _parse_date(date_str: str) -> datetime:
    try:
        return datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)


def detect_trends(
    clusters: list[TopicCluster],
    trend_log: TrendLog,
    date: str,
) -> list[Trend]:
    """Compare current clusters against historical data."""
    # Current topic counts
    current = {c.topic: len(c.entries) for c in clusters}

    # Record this run
    trend_log.append_run(date, current)

    # Compare against previous window
    previous = trend_log.previous_counts()

    if not previous:
        return []  # First run, no trends yet

    trends = []
    all_topics = set(current.keys()) | set(previous.keys())

    for topic in all_topics:
        cur = current.get(topic, 0)
        prev = previous.get(topic, 0)

        if cur == 0:
            continue  # Topic disappeared — not interesting for the briefing

        if prev == 0:
            trends.append(Trend(topic, cur, 0, 'new'))
        elif cur > prev:
            trends.append(Trend(topic, cur, prev, 'up'))
        elif cur < prev:
            trends.append(Trend(topic, cur, prev, 'down'))
        else:
            trends.append(Trend(topic, cur, prev, 'stable'))

    # Sort: new and up first
    order = {'new': 0, 'up': 1, 'down': 2, 'stable': 3}
    trends.sort(key=lambda t: (order.get(t.direction, 9), -t.current_count))

    return trends


# =============================================================================
# Core Tapestry Builder
# =============================================================================

async def weave_tapestry(
    playlist_urls: list[str],
    batch_size: int = 20,
    checkpoint_dir: Path | None = None,
    summarize: bool = True,
) -> Tapestry:
    """
    Scan playlists and build a tapestry (structured briefing).

    Pipeline:
      1. PlaylistSpinner fetches manifests + transcripts
      2. LLM summarizes each transcript (if enabled)
      3. Clustering groups entries by topic
      4. Trend detection compares against history

    Args:
        playlist_urls: YouTube playlist URLs to scan.
        batch_size: Max videos per run.
        checkpoint_dir: Checkpoint persistence directory.
        summarize: Whether to run LLM summarization.

    Returns:
        Tapestry with entries, clusters, trends, and resonance highlights.
    """
    checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch and ingest
    spinner = PlaylistSpinner(
        importance_threshold=0.0,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
    )

    result = await spinner.spin(playlist_urls)

    # Step 2: Build entries with optional summarization
    entries = []
    for shard in result.shards:
        meta = shard.metadata
        title = meta.get('title', 'Unknown')

        if summarize and meta.get('shard_type') == 'transcript':
            summary = await summarize_transcript(shard.text, title)
        else:
            summary = shard.text[:200] + ('...' if len(shard.text) > 200 else '')

        entries.append(TapestryEntry(
            video_id=meta.get('video_id', shard.id),
            title=title,
            channel=meta.get('channel', 'Unknown'),
            url=meta.get('url', ''),
            resonance_count=meta.get('resonance_count', 1),
            importance=meta.get('importance_score', 0.5),
            summary=summary,
            playlists=meta.get('playlists', []),
            shard_type=meta.get('shard_type', 'transcript'),
        ))

    # Step 3: Separate high-resonance
    high_resonance = sorted(
        [e for e in entries if e.resonance_count >= 2],
        key=lambda e: e.resonance_count,
        reverse=True,
    )

    # Step 4: Cluster by topic
    clusters = cluster_entries(entries)

    # Step 5: Trend detection
    date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    trend_log_path = checkpoint_dir / TREND_LOG_FILE
    trend_log = TrendLog.load(trend_log_path)
    trends = detect_trends(clusters, trend_log, date)
    trend_log.save(trend_log_path)

    total_scanned = len(spinner.resonance_index.all_video_ids())

    tapestry = Tapestry(
        date=date,
        total_videos_scanned=total_scanned,
        new_videos_processed=len(entries),
        clusters=clusters,
        high_resonance=high_resonance,
        trends=trends,
        skipped_count=max(0, len(result.warnings)),
    )

    return tapestry


# =============================================================================
# Phase 4: Delivery — Vault Federation
# =============================================================================

async def propose_to_vault(tapestry: Tapestry):
    """Propose the tapestry as a note to the PARA vault."""
    try:
        from hololoom.apps.server.vault_bridge import propose_note
        await propose_note(
            title=f"Daily Tapestry -- {tapestry.date}",
            content=tapestry.to_markdown(),
            links=[],
            agent_id='tapestry',
        )
        logger.info("Proposed tapestry to vault")
    except ImportError:
        logger.warning("Vault bridge unavailable -- skipping vault proposal")
    except Exception as e:
        logger.warning("Failed to propose to vault: %s", e)


# =============================================================================
# Phase 4: Delivery — Matrix via Femtoclaw IPC
# =============================================================================

def deliver_to_matrix(tapestry: Tapestry, room_jid: str):
    """
    Send the tapestry to a Matrix room via Femtoclaw's file-based IPC.

    Drops a JSON message in the main group's IPC directory.
    The Femtoclaw IPC watcher picks it up and calls sendMessage().
    No HTTP needed — pure filesystem protocol.
    """
    ipc_dir = FEMTOCLAW_IPC_DIR / 'main' / 'messages'
    ipc_dir.mkdir(parents=True, exist_ok=True)

    filename = f"tapestry-{int(time.time() * 1000)}.json"
    payload = {
        'type': 'message',
        'chatJid': room_jid,
        'text': tapestry.to_markdown(),
    }

    # Atomic write: write to .tmp then rename
    tmp_path = ipc_dir / f"{filename}.tmp"
    final_path = ipc_dir / filename

    with open(tmp_path, 'w') as f:
        json.dump(payload, f)
    tmp_path.rename(final_path)

    logger.info("Dropped tapestry IPC message for room %s", room_jid)


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description='Daily Tapestry -- YouTube playlist briefing',
    )
    parser.add_argument('--playlists', type=Path, help='Path to playlists.txt file')
    parser.add_argument('--batch-size', type=int, default=20, help='Max videos per run')
    parser.add_argument('--checkpoint-dir', type=Path, default=None, help='Checkpoint directory')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--vault', action='store_true', help='Propose to PARA vault')
    parser.add_argument(
        '--matrix', type=str, metavar='ROOM_JID',
        help='Send to Matrix room via Femtoclaw IPC (e.g. !abc:piserver.local)',
    )
    parser.add_argument(
        '--no-summarize', action='store_true',
        help='Skip LLM summarization (faster, raw text only)',
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Load playlist URLs
    urls = load_playlist_urls(args.playlists)
    if not urls:
        print("No playlist URLs configured.")
        print(f"Create {DEFAULT_PLAYLIST_FILE} with one playlist URL per line.")
        return

    # Weave the tapestry
    tapestry = await weave_tapestry(
        playlist_urls=urls,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        summarize=not args.no_summarize,
    )

    # Output
    if args.json:
        print(json.dumps(tapestry.to_dict(), indent=2))
    else:
        print(tapestry.to_markdown())

    # Delivery routes
    if args.vault:
        await propose_to_vault(tapestry)

    if args.matrix:
        deliver_to_matrix(tapestry, args.matrix)


if __name__ == '__main__':
    asyncio.run(main())
