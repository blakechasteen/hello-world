# -*- coding: utf-8 -*-
"""
Org-Mode Live File Monitoring
==============================
Real-time file watching with incremental YarnGraph updates and temporal evolution.

Features:
- Watchdog-based file monitoring (inotify on Linux, FSEvents on macOS)
- Incremental updates to YarnGraph (only changed headings)
- Temporal evolution tracking (preserves history of changes)
- Change detection and diffing
- Automatic graph synchronization

Design Philosophy:
- Org-mode remains the source of truth
- HoloLoom mirrors org structure in real-time
- Track not just current state, but evolution over time
- Enable temporal queries: "When did this TODO change to DONE?"

Example:
```python
from HoloLoom.spinningWheel.orgmode_live import OrgLiveMonitor
from HoloLoom.memory.graph import KG

kg = KG()
monitor = OrgLiveMonitor(kg, watch_dir='~/org')

# Start monitoring
await monitor.start()

# Files are automatically synced to graph
# Changes trigger incremental updates
# History is preserved in temporal edges

await monitor.stop()
```
"""

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable, Any
import logging

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None

from .orgmode import OrgModeSpinner, OrgHeading
from .base import SpinnerConfig

try:
    from HoloLoom.memory.graph import KG, KGEdge
    from HoloLoom.Documentation.types import MemoryShard
except ImportError:
    # Fallback for standalone testing
    KG = None
    KGEdge = None
    MemoryShard = None

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ChangeEvent:
    """
    Record of a change to an org-mode heading.

    Used for temporal evolution tracking.
    """
    heading_id: str                         # Org heading ID
    filepath: str                           # Source file
    timestamp: datetime                     # When change occurred
    change_type: str                        # "created", "modified", "deleted", "todo_state_change"
    old_value: Optional[Any] = None         # Previous value (for modifications)
    new_value: Optional[Any] = None         # New value
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeadingSnapshot:
    """
    Snapshot of a heading at a point in time.

    Used for change detection and diffing.
    """
    heading_id: str
    content_hash: str                       # Hash of heading content for change detection
    title: str
    todo_state: Optional[str]
    tags: List[str]
    level: int
    parent_id: Optional[str]
    properties: Dict[str, str]
    last_modified: datetime


# ============================================================================
# Live File Watcher
# ============================================================================

class OrgFileEventHandler(FileSystemEventHandler):
    """
    Watchdog event handler for org-mode files.

    Triggers callbacks when .org files are created or modified.
    """

    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self._debounce_cache: Dict[str, float] = {}
        self._debounce_delay = 0.5  # seconds

    def on_modified(self, event):
        if event.is_directory:
            return

        filepath = event.src_path
        if not filepath.endswith('.org'):
            return

        # Debounce: avoid multiple rapid calls for same file
        import time
        now = time.time()
        last_call = self._debounce_cache.get(filepath, 0)

        if now - last_call < self._debounce_delay:
            return

        self._debounce_cache[filepath] = now
        self.callback(filepath)

    def on_created(self, event):
        if event.is_directory:
            return

        filepath = event.src_path
        if filepath.endswith('.org'):
            self.callback(filepath)


# ============================================================================
# Live Monitor with Incremental Updates
# ============================================================================

class OrgLiveMonitor:
    """
    Live file monitor with incremental YarnGraph updates and temporal evolution.

    Features:
    - Real-time file watching (watchdog/inotify)
    - Change detection (content hashing)
    - Incremental graph updates (only changed headings)
    - Temporal evolution tracking (history preservation)
    - Automatic synchronization

    Architecture:
    1. Watch org files for changes
    2. Detect which headings changed (content hash comparison)
    3. Update only changed shards in memory
    4. Update YarnGraph edges incrementally
    5. Record change events for temporal queries

    Usage:
    ```python
    kg = KG()
    monitor = OrgLiveMonitor(kg, watch_dir='~/org')

    await monitor.start()
    # ... do other work ...
    await monitor.stop()
    ```
    """

    def __init__(
        self,
        yarn_graph: 'KG',
        watch_dir: Optional[str] = None,
        watch_files: Optional[List[str]] = None,
        config: Optional[SpinnerConfig] = None,
        enable_temporal_evolution: bool = True
    ):
        """
        Initialize live monitor.

        Args:
            yarn_graph: KG instance to update
            watch_dir: Directory to watch (all .org files)
            watch_files: Specific files to watch (alternative to watch_dir)
            config: SpinnerConfig for parsing
            enable_temporal_evolution: Track change history
        """
        if not WATCHDOG_AVAILABLE:
            raise ImportError(
                "watchdog is required for live monitoring. "
                "Install with: pip install watchdog"
            )

        self.yarn_graph = yarn_graph
        self.watch_dir = Path(watch_dir).expanduser() if watch_dir else None
        self.watch_files = [Path(f).expanduser() for f in watch_files] if watch_files else []
        self.config = config or SpinnerConfig(enable_enrichment=False)
        self.enable_temporal_evolution = enable_temporal_evolution

        # State tracking
        self.spinner = OrgModeSpinner(self.config)
        self.snapshots: Dict[str, HeadingSnapshot] = {}  # heading_id -> snapshot
        self.file_snapshots: Dict[str, Dict[str, HeadingSnapshot]] = {}  # filepath -> {heading_id -> snapshot}
        self.change_history: List[ChangeEvent] = []

        # Watchdog
        self.observer: Optional[Observer] = None
        self.running = False

        # Callbacks
        self.on_change_callbacks: List[Callable[[ChangeEvent], None]] = []

        # Logger
        self.logger = logging.getLogger(__name__)

    async def start(self):
        """Start monitoring files."""
        if not WATCHDOG_AVAILABLE:
            raise RuntimeError("watchdog not available")

        self.running = True

        # Initial load of all files
        await self._initial_load()

        # Start watchdog observer
        self.observer = Observer()
        event_handler = OrgFileEventHandler(callback=self._on_file_change)

        if self.watch_dir:
            self.observer.schedule(event_handler, str(self.watch_dir), recursive=True)
            self.logger.info(f"Watching directory: {self.watch_dir}")

        for filepath in self.watch_files:
            if filepath.is_file():
                # Watch parent directory but filter in callback
                self.observer.schedule(event_handler, str(filepath.parent), recursive=False)
                self.logger.info(f"Watching file: {filepath}")

        self.observer.start()
        self.logger.info("Live monitoring started")

    async def stop(self):
        """Stop monitoring."""
        self.running = False

        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None

        self.logger.info("Live monitoring stopped")

    async def _initial_load(self):
        """Load all watched files initially."""
        files_to_load = []

        if self.watch_dir:
            files_to_load.extend(self.watch_dir.rglob('*.org'))

        files_to_load.extend(self.watch_files)

        for filepath in files_to_load:
            if filepath.is_file():
                await self._load_file(str(filepath), is_initial=True)

    def _on_file_change(self, filepath: str):
        """Callback when file changes (called by watchdog)."""
        # Run async handler in event loop
        asyncio.create_task(self._handle_file_change(filepath))

    async def _handle_file_change(self, filepath: str):
        """Handle file change asynchronously."""
        try:
            await self._load_file(filepath, is_initial=False)
        except Exception as e:
            self.logger.error(f"Error processing {filepath}: {e}")

    async def _load_file(self, filepath: str, is_initial: bool = False):
        """
        Load/reload org file and update graph incrementally.

        Args:
            filepath: Path to org file
            is_initial: True if initial load, False if update
        """
        filepath = str(Path(filepath).resolve())

        # Parse file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        shards = await self.spinner.spin({
            'content': content,
            'source': filepath,
            'episode': f'org:{Path(filepath).name}'
        })

        # Create snapshots for current state
        new_snapshots = {}
        for shard in shards:
            snapshot = self._shard_to_snapshot(shard)
            new_snapshots[snapshot.heading_id] = snapshot

        # Get old snapshots for this file
        old_snapshots = self.file_snapshots.get(filepath, {})

        if is_initial:
            # Initial load: add all to graph
            self.logger.info(f"Initial load: {filepath} ({len(shards)} headings)")
            self._add_shards_to_graph(shards)
        else:
            # Incremental update: detect changes
            changes = self._detect_changes(old_snapshots, new_snapshots, filepath)

            if changes:
                self.logger.info(f"Detected {len(changes)} changes in {filepath}")
                await self._apply_changes(changes, shards)
            else:
                self.logger.debug(f"No changes in {filepath}")

        # Update snapshot cache
        self.file_snapshots[filepath] = new_snapshots
        self.snapshots.update(new_snapshots)

    def _shard_to_snapshot(self, shard: MemoryShard) -> HeadingSnapshot:
        """Convert MemoryShard to HeadingSnapshot."""
        # Create content hash for change detection
        content_str = f"{shard.text}:{','.join(sorted(shard.motifs))}"
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]

        # Extract TODO state from motifs
        todo_state = None
        for motif in shard.motifs:
            if motif.startswith('TODO:'):
                todo_state = motif.split(':')[1]
                break

        # Extract tags from motifs
        tags = [m.split(':')[1] for m in shard.motifs if m.startswith('TAG:')]

        return HeadingSnapshot(
            heading_id=shard.id,
            content_hash=content_hash,
            title=shard.entities[0] if shard.entities else shard.id,
            todo_state=todo_state,
            tags=tags,
            level=shard.metadata.get('level', 1),
            parent_id=shard.metadata.get('parent_id'),
            properties=shard.metadata.get('properties', {}),
            last_modified=datetime.now()
        )

    def _detect_changes(
        self,
        old_snapshots: Dict[str, HeadingSnapshot],
        new_snapshots: Dict[str, HeadingSnapshot],
        filepath: str
    ) -> List[ChangeEvent]:
        """
        Detect what changed between old and new snapshots.

        Returns list of ChangeEvents.
        """
        changes = []
        timestamp = datetime.now()

        old_ids = set(old_snapshots.keys())
        new_ids = set(new_snapshots.keys())

        # Deleted headings
        for heading_id in old_ids - new_ids:
            changes.append(ChangeEvent(
                heading_id=heading_id,
                filepath=filepath,
                timestamp=timestamp,
                change_type='deleted',
                old_value=old_snapshots[heading_id].title,
                new_value=None
            ))

        # Created headings
        for heading_id in new_ids - old_ids:
            changes.append(ChangeEvent(
                heading_id=heading_id,
                filepath=filepath,
                timestamp=timestamp,
                change_type='created',
                old_value=None,
                new_value=new_snapshots[heading_id].title
            ))

        # Modified headings
        for heading_id in old_ids & new_ids:
            old_snap = old_snapshots[heading_id]
            new_snap = new_snapshots[heading_id]

            # Check if content changed
            if old_snap.content_hash != new_snap.content_hash:
                # Check specific change types
                if old_snap.todo_state != new_snap.todo_state:
                    changes.append(ChangeEvent(
                        heading_id=heading_id,
                        filepath=filepath,
                        timestamp=timestamp,
                        change_type='todo_state_change',
                        old_value=old_snap.todo_state,
                        new_value=new_snap.todo_state,
                        metadata={'title': new_snap.title}
                    ))
                else:
                    changes.append(ChangeEvent(
                        heading_id=heading_id,
                        filepath=filepath,
                        timestamp=timestamp,
                        change_type='modified',
                        old_value=old_snap.content_hash,
                        new_value=new_snap.content_hash
                    ))

        return changes

    async def _apply_changes(self, changes: List[ChangeEvent], new_shards: List[MemoryShard]):
        """
        Apply changes to YarnGraph incrementally.

        Only updates changed headings, preserving the rest of the graph.
        """
        for change in changes:
            if self.enable_temporal_evolution:
                self._record_change(change)

            # Find corresponding shard
            shard = next((s for s in new_shards if s.id == change.heading_id), None)

            if change.change_type == 'deleted':
                # Remove from graph (optional - could keep for history)
                self._remove_from_graph(change.heading_id)

            elif change.change_type == 'created' and shard:
                # Add new heading to graph
                self._add_shards_to_graph([shard])

            elif change.change_type in ('modified', 'todo_state_change') and shard:
                # Update existing heading
                self._update_shard_in_graph(shard)

            # Trigger callbacks
            for callback in self.on_change_callbacks:
                try:
                    callback(change)
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")

        self.logger.info(f"Applied {len(changes)} changes to YarnGraph")

    def _add_shards_to_graph(self, shards: List[MemoryShard]):
        """Add shards to YarnGraph."""
        for shard in shards:
            # Add entity node
            entity_id = shard.id

            # Add hierarchical edge (parent-child)
            if shard.metadata.get('parent_id'):
                parent_id = shard.metadata['parent_id']
                self.yarn_graph.add_edge(KGEdge(
                    src=entity_id,
                    dst=parent_id,
                    type='CHILD_OF',
                    weight=1.0,
                    metadata={'level': shard.metadata.get('level', 1)}
                ))

            # Add links as edges
            if 'links' in shard.metadata:
                for link in shard.metadata['links']:
                    self.yarn_graph.add_edge(KGEdge(
                        src=entity_id,
                        dst=link['target'],
                        type='LINKS_TO',
                        weight=1.0,
                        metadata={'description': link['description']}
                    ))

            # Add temporal edge if deadline/scheduled
            if shard.metadata.get('deadline'):
                deadline = shard.metadata['deadline']
                self.yarn_graph.add_edge(KGEdge(
                    src=entity_id,
                    dst=f"time::{deadline}",
                    type='DEADLINE',
                    weight=1.0
                ))

            if shard.metadata.get('scheduled'):
                scheduled = shard.metadata['scheduled']
                self.yarn_graph.add_edge(KGEdge(
                    src=entity_id,
                    dst=f"time::{scheduled}",
                    type='SCHEDULED',
                    weight=1.0
                ))

    def _update_shard_in_graph(self, shard: MemoryShard):
        """Update existing shard in graph (re-add edges)."""
        # For simplicity, remove and re-add
        # In production, could be more selective
        self._remove_from_graph(shard.id)
        self._add_shards_to_graph([shard])

    def _remove_from_graph(self, heading_id: str):
        """Remove heading from graph."""
        # Remove all edges involving this node
        if heading_id in self.yarn_graph.G:
            self.yarn_graph.G.remove_node(heading_id)

    def _record_change(self, change: ChangeEvent):
        """Record change event for temporal evolution."""
        self.change_history.append(change)

        # Add temporal evolution edge to graph
        timestamp_str = change.timestamp.isoformat()

        self.yarn_graph.add_edge(KGEdge(
            src=change.heading_id,
            dst=f"change::{timestamp_str}",
            type=f'CHANGE_{change.change_type.upper()}',
            weight=1.0,
            metadata={
                'old_value': str(change.old_value) if change.old_value else None,
                'new_value': str(change.new_value) if change.new_value else None,
                'filepath': change.filepath,
                **change.metadata
            }
        ))

    def on_change(self, callback: Callable[[ChangeEvent], None]):
        """
        Register callback for change events.

        Example:
        ```python
        def handle_change(event: ChangeEvent):
            if event.change_type == 'todo_state_change':
                print(f"{event.metadata['title']}: {event.old_value} -> {event.new_value}")

        monitor.on_change(handle_change)
        ```
        """
        self.on_change_callbacks.append(callback)

    def get_change_history(
        self,
        heading_id: Optional[str] = None,
        change_type: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[ChangeEvent]:
        """
        Query change history.

        Args:
            heading_id: Filter by heading ID
            change_type: Filter by change type
            since: Filter by timestamp (changes after this time)

        Returns:
            Filtered list of ChangeEvents
        """
        filtered = self.change_history

        if heading_id:
            filtered = [c for c in filtered if c.heading_id == heading_id]

        if change_type:
            filtered = [c for c in filtered if c.change_type == change_type]

        if since:
            filtered = [c for c in filtered if c.timestamp >= since]

        return filtered

    def get_todo_transitions(self) -> Dict[str, List[tuple]]:
        """
        Get TODO state transitions for all headings.

        Returns:
            Dict mapping heading_id -> list of (timestamp, old_state, new_state)
        """
        transitions = {}

        for change in self.change_history:
            if change.change_type == 'todo_state_change':
                if change.heading_id not in transitions:
                    transitions[change.heading_id] = []

                transitions[change.heading_id].append((
                    change.timestamp,
                    change.old_value,
                    change.new_value
                ))

        return transitions


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("OrgLiveMonitor requires running in async context")
    print("See demo_orgmode_live.py for usage examples")
