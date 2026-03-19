"""Tests for ThresholdNotifier — alert mechanism for scored observations."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import json
import pytest

from hive.notifier import ThresholdNotifier


class TestThresholdNotifier:
    def test_p0_triggers_alert(self):
        n = ThresholdNotifier()
        alert = n.check_and_alert("node-1", {"band": "P0", "composite": 85.0}, "varroa emergency")
        assert alert is not None
        assert alert["band"] == "P0"
        assert alert["node_id"] == "node-1"
        assert n.pending_count == 1

    def test_p1_triggers_alert(self):
        n = ThresholdNotifier()
        alert = n.check_and_alert("node-2", {"band": "P1", "composite": 45.0}, "stores low")
        assert alert is not None
        assert alert["band"] == "P1"

    def test_p2_does_not_trigger(self):
        n = ThresholdNotifier()
        alert = n.check_and_alert("node-3", {"band": "P2", "composite": 20.0}, "monitor")
        assert alert is None
        assert n.pending_count == 0

    def test_p3_does_not_trigger(self):
        n = ThresholdNotifier()
        alert = n.check_and_alert("node-4", {"band": "P3", "composite": 5.0}, "info")
        assert alert is None

    def test_custom_alert_bands(self):
        n = ThresholdNotifier(alert_bands={"P0"})
        assert n.check_and_alert("a", {"band": "P1"}, "") is None
        assert n.check_and_alert("b", {"band": "P0"}, "") is not None

    def test_callback_fires(self):
        fired = []
        n = ThresholdNotifier(callbacks=[lambda a: fired.append(a)])
        n.check_and_alert("x", {"band": "P0", "composite": 90.0}, "test")
        assert len(fired) == 1
        assert fired[0]["band"] == "P0"

    def test_callback_failure_does_not_crash(self):
        def bad_cb(a):
            raise RuntimeError("boom")
        n = ThresholdNotifier(callbacks=[bad_cb])
        alert = n.check_and_alert("x", {"band": "P0"}, "test")
        assert alert is not None  # Should still return alert despite callback failure

    def test_summary(self):
        n = ThresholdNotifier()
        n.check_and_alert("a", {"band": "P0", "composite": 90}, "")
        n.check_and_alert("b", {"band": "P0", "composite": 85}, "")
        n.check_and_alert("c", {"band": "P1", "composite": 40}, "")
        s = n.summary()
        assert "2 P0" in s
        assert "1 P1" in s

    def test_summary_no_alerts(self):
        n = ThresholdNotifier()
        assert n.summary() == "No alerts"

    def test_hook_store_integration(self):
        """Verify hook_store works with StoredItem-like objects."""
        n = ThresholdNotifier()

        class FakeStored:
            content = "varroa count critical"
            properties = {"band": "P0", "composite": 95.0, "status": "open"}

        n.hook_store("node-1", FakeStored())
        assert n.pending_count == 1


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "alerts.json"
        n = ThresholdNotifier(alerts_path=path)
        n.check_and_alert("a", {"band": "P0", "composite": 90}, "test alert")
        n.save_alerts()

        loaded = n.load_alerts()
        assert len(loaded) == 1
        assert loaded[0]["band"] == "P0"

    def test_unacknowledged(self, tmp_path):
        path = tmp_path / "alerts.json"
        n = ThresholdNotifier(alerts_path=path)
        n.check_and_alert("a", {"band": "P0", "composite": 90}, "test")
        n.save_alerts()

        unack = n.unacknowledged()
        assert len(unack) == 1

    def test_acknowledge(self, tmp_path):
        path = tmp_path / "alerts.json"
        n = ThresholdNotifier(alerts_path=path)
        n.check_and_alert("a", {"band": "P0", "composite": 90}, "test")
        n.save_alerts()

        assert n.acknowledge("a") is True
        unack = n.unacknowledged()
        assert len(unack) == 0

    def test_acknowledge_nonexistent(self, tmp_path):
        path = tmp_path / "alerts.json"
        n = ThresholdNotifier(alerts_path=path)
        assert n.acknowledge("nope") is False

    def test_appends_to_existing(self, tmp_path):
        path = tmp_path / "alerts.json"
        # Pre-populate
        path.write_text(json.dumps([{"node_id": "old", "band": "P0", "acknowledged": True}]))

        n = ThresholdNotifier(alerts_path=path)
        n.check_and_alert("new", {"band": "P0", "composite": 80}, "new alert")
        n.save_alerts()

        all_alerts = n.load_alerts()
        assert len(all_alerts) == 2
        assert all_alerts[0]["node_id"] == "old"
        assert all_alerts[1]["node_id"] == "new"
