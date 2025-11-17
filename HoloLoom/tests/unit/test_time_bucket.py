from datetime import datetime, timezone, timedelta

from HoloLoom.Utils.time_bucket import time_bucket, to_utc_datetime
from HoloLoom.memory.graph import KG


def test_time_bucket_from_iso_string():
    bucket = time_bucket("2024-03-10T16:30:00Z")
    assert bucket == "2024-03-10-afternoon"


def test_time_bucket_from_epoch_seconds():
    ts = datetime(2024, 3, 10, 23, 45, tzinfo=timezone.utc).timestamp()
    bucket = time_bucket(ts)
    assert bucket == "2024-03-10-night"


def test_to_utc_datetime_normalises_inputs():
    naive = datetime(2024, 3, 10, 6, 0, 0)
    aware = to_utc_datetime(naive)
    assert aware.tzinfo == timezone.utc
    assert aware.hour == 6

    offset = datetime(2024, 3, 10, 21, 0, 0, tzinfo=timezone(timedelta(hours=-5)))
    aware_offset = to_utc_datetime(offset)
    assert aware_offset.hour == 2  # converted to UTC
    assert aware_offset.tzinfo == timezone.utc


def test_connect_entity_to_time_creates_thread_and_edge():
    kg = KG()
    thread_id = kg.connect_entity_to_time("episode:1", "2024-03-10T22:15:00Z")

    assert thread_id == "time::2024-03-10-night"
    assert thread_id in kg.G
    assert kg.G.nodes[thread_id]["bucket"] == "2024-03-10-night"

    assert ("episode:1", thread_id) in kg.G.edges()
    edge_data = list(kg.G.get_edge_data("episode:1", thread_id).values())[0]
    assert edge_data["type"] == "IN_TIME"
    assert edge_data["bucket"] == "2024-03-10-night"
    assert edge_data["timestamp"].startswith("2024-03-10T22:15:00")

    # entity index updated both ways
    assert thread_id in kg._entity_index["episode:1"]
    assert "episode:1" in kg._entity_index[thread_id]


def test_connect_entity_to_time_reuses_thread_node():
    kg = KG()
    kg.connect_entity_to_time("episode:1", "2024-03-10T09:00:00Z")
    kg.connect_entity_to_time("episode:2", "2024-03-10T11:59:59Z")

    thread_id = "time::2024-03-10-morning"
    assert thread_id in kg.G
    assert "episode:1" in kg._entity_index[thread_id]
    assert "episode:2" in kg._entity_index[thread_id]
    assert kg.G.number_of_nodes() == 3  # two entities + one time thread

