"""ThresholdNotifier — alerts when scored observations cross priority thresholds.

Works in two modes:
- API mode: POSTs to HoloLoom's signal endpoint
- Local mode: appends to ~/.hive/alerts.json

Designed as a post-store hook for the bus or callable from jobs directly.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ThresholdNotifier:
    """Fires alerts when stored observations exceed a threshold band.

    Usage:
        notifier = ThresholdNotifier(alert_bands={"P0", "P1"})

        # As post-store hook (local bus):
        bus._post_store_hooks.append(notifier.hook_store)

        # Or call directly after storing:
        notifier.check_and_alert(node_id, obs.properties)
    """

    def __init__(
        self,
        alert_bands: Optional[set] = None,
        alerts_path: Optional[Path] = None,
        api_url: Optional[str] = None,
        callbacks: Optional[List[Callable]] = None,
    ):
        """
        Args:
            alert_bands: Set of bands that trigger alerts. Default: {"P0", "P1"}.
            alerts_path: Path to alerts JSON file. Default: ~/.hive/alerts.json.
            api_url: HoloLoom API URL for signal emission (optional).
            callbacks: Additional callables to invoke on alert: fn(alert_dict).
        """
        self.alert_bands = alert_bands or {"P0", "P1"}
        self.alerts_path = alerts_path or Path.home() / ".hive" / "alerts.json"
        self.api_url = api_url
        self.callbacks = callbacks or []
        self._alerts: List[Dict[str, Any]] = []

    def check_and_alert(
        self,
        node_id: str,
        properties: Dict[str, Any],
        content: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Check if an observation's band crosses the alert threshold.

        Returns the alert dict if triggered, None otherwise.
        """
        band = properties.get("band")
        if band not in self.alert_bands:
            return None

        alert = {
            "node_id": node_id,
            "band": band,
            "content": content[:200],
            "composite": properties.get("composite", 0),
            "mode": properties.get("mode", "unknown"),
            "status": properties.get("status", "open"),
            "alerted_at": datetime.now(timezone.utc).isoformat(),
            "acknowledged": False,
        }

        self._alerts.append(alert)
        logger.warning("ALERT [%s]: %s (score=%.1f)", band, content[:80], alert["composite"])

        # Fire callbacks
        for cb in self.callbacks:
            try:
                cb(alert)
            except Exception as e:
                logger.warning("Alert callback failed: %s", e)

        return alert

    def hook_store(self, item_id: str, stored_item: Any) -> None:
        """Post-store hook for LiteMemoryBus. Checks properties and alerts."""
        props = getattr(stored_item, "properties", None)
        if not props:
            return
        content = getattr(stored_item, "content", "")
        self.check_and_alert(item_id, props, content)

    def save_alerts(self, path: Optional[Path] = None) -> None:
        """Persist alerts to JSON (append to existing)."""
        path = path or self.alerts_path
        path.parent.mkdir(parents=True, exist_ok=True)

        existing = []
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                existing = []

        combined = existing + self._alerts
        # Keep last 100 alerts
        combined = combined[-100:]

        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(combined, indent=2))
        tmp.replace(path)
        logger.debug("Saved %d alerts to %s", len(combined), path)

    def load_alerts(self, path: Optional[Path] = None) -> List[Dict]:
        """Load alerts from JSON."""
        path = path or self.alerts_path
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text())
        except Exception:
            return []

    def unacknowledged(self, path: Optional[Path] = None) -> List[Dict]:
        """Get unacknowledged alerts."""
        alerts = self.load_alerts(path)
        return [a for a in alerts if not a.get("acknowledged")]

    def acknowledge(self, node_id: str, path: Optional[Path] = None) -> bool:
        """Mark an alert as acknowledged. Returns True if found."""
        path = path or self.alerts_path
        alerts = self.load_alerts(path)
        found = False
        for a in alerts:
            if a.get("node_id") == node_id:
                a["acknowledged"] = True
                found = True
        if found:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(alerts, indent=2))
            tmp.replace(path)
        return found

    @property
    def pending_count(self) -> int:
        """Number of alerts fired in this session."""
        return len(self._alerts)

    def summary(self) -> str:
        """One-line summary of alerts fired this session."""
        if not self._alerts:
            return "No alerts"
        bands = {}
        for a in self._alerts:
            b = a["band"]
            bands[b] = bands.get(b, 0) + 1
        parts = [f"{count} {band}" for band, count in sorted(bands.items())]
        return f"ALERTS: {', '.join(parts)}"
