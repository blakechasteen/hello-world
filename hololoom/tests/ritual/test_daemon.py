"""Tests for daemon protocol and skeleton."""

import json
import pytest

from hololoom.core.runtime.daemon_protocol import (
    DaemonRequest,
    DaemonResponse,
    DaemonEvent,
)
from hololoom.core.runtime.daemon import HoloLoomDaemon
from hololoom.core.runtime import HoloLoomRuntime
from hololoom.core.runtime.ritual_layer import RitualLayer


class TestDaemonProtocol:
    def test_request_roundtrip(self):
        req = DaemonRequest(type="status", payload={"verbose": True})
        json_str = req.to_json()
        req2 = DaemonRequest.from_json(json_str)
        assert req2.type == "status"
        assert req2.payload["verbose"] is True

    def test_response_roundtrip(self):
        resp = DaemonResponse(
            type="result",
            thread_id="t-1",
            payload={"layers": {"ritual": {"state": "running"}}},
        )
        json_str = resp.to_json()
        resp2 = DaemonResponse.from_json(json_str)
        assert resp2.type == "result"
        assert resp2.payload["layers"]["ritual"]["state"] == "running"

    def test_event_roundtrip(self):
        evt = DaemonEvent(
            thread_id="t-1",
            signal={"kind": "heartbeat", "tick": 42},
        )
        json_str = evt.to_json()
        evt2 = DaemonEvent.from_json(json_str)
        assert evt2.signal["kind"] == "heartbeat"
        assert evt2.signal["tick"] == 42

    def test_error_response(self):
        resp = DaemonResponse.error("something broke", request_id="req-1")
        assert resp.type == "error"
        assert resp.payload["error"] == "something broke"
        assert resp.request_id == "req-1"


class TestDaemonRequestValidation:
    def test_valid_status(self):
        valid, err = DaemonRequest.validate('{"type": "status"}')
        assert valid
        assert err == ""

    def test_valid_query(self):
        valid, err = DaemonRequest.validate('{"type": "query", "payload": {"text": "hello"}}')
        assert valid

    def test_invalid_json(self):
        valid, err = DaemonRequest.validate("not json")
        assert not valid
        assert "Invalid JSON" in err

    def test_missing_type(self):
        valid, err = DaemonRequest.validate('{"payload": {}}')
        assert not valid
        assert "type" in err

    def test_unknown_type(self):
        valid, err = DaemonRequest.validate('{"type": "destroy_everything"}')
        assert not valid
        assert "Unknown type" in err

    def test_not_object(self):
        valid, err = DaemonRequest.validate('"just a string"')
        assert not valid


class TestHoloLoomDaemon:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        daemon = HoloLoomDaemon()
        await daemon.start()
        assert daemon.is_running
        await daemon.stop()
        assert not daemon.is_running

    @pytest.mark.asyncio
    async def test_handle_status(self):
        rt = HoloLoomRuntime()
        layer = RitualLayer(register_defaults=False)
        rt.register_layer(layer)

        daemon = HoloLoomDaemon(runtime=rt)
        await daemon.start()

        req = DaemonRequest(type="status")
        resp = await daemon.handle_request(req)

        assert resp.type == "result"
        assert "layers" in resp.payload
        assert "ritual" in resp.payload["layers"]
        assert resp.payload["layers"]["ritual"]["state"] == "running"

        await daemon.stop()

    @pytest.mark.asyncio
    async def test_handle_unknown_type(self):
        daemon = HoloLoomDaemon()
        await daemon.start()

        req = DaemonRequest(type="nonexistent_handler")
        resp = await daemon.handle_request(req)
        assert resp.type == "error"

        await daemon.stop()

    @pytest.mark.asyncio
    async def test_handle_query_stub(self):
        daemon = HoloLoomDaemon()
        await daemon.start()

        req = DaemonRequest(type="query", payload={"text": "hello"})
        resp = await daemon.handle_request(req)
        assert resp.type == "result"
        assert "not yet wired" in resp.payload.get("message", "")

        await daemon.stop()

    @pytest.mark.asyncio
    async def test_runtime_accessible(self):
        rt = HoloLoomRuntime()
        daemon = HoloLoomDaemon(runtime=rt)
        assert daemon.runtime is rt
