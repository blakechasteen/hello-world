"""
Simple standalone tests for Chronos Core (no pytest required)
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chronos.core import ChronosState, EventLog, ChronosEvent


def test_start_stop():
    """Test basic start/stop cycle"""
    print("Testing START/STOP...", end=" ")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        log_path = Path(f.name)

    try:
        state = ChronosState(log_path)

        # Start task
        msg = state.start("test_task", tags=["test"])
        assert "Started test_task" in msg
        assert state.active_task is not None

        # Stop task
        msg = state.stop()
        assert "Stopped test_task" in msg
        assert state.active_task is None

        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False
    finally:
        if log_path.exists():
            log_path.unlink()


def test_log_retroactive():
    """Test logging completed tasks"""
    print("Testing LOG (retroactive)...", end=" ")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        log_path = Path(f.name)

    try:
        state = ChronosState(log_path)

        msg = state.log("email_review", duration_sec=1200, tags=["admin"])
        assert "Logged email_review" in msg
        assert "20m" in msg

        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False
    finally:
        if log_path.exists():
            log_path.unlink()


def test_note():
    """Test adding notes"""
    print("Testing NOTE...", end=" ")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        log_path = Path(f.name)

    try:
        state = ChronosState(log_path)

        # Start task
        state.start("test_task")

        # Add note
        msg = state.note("Test note content")
        assert "Note added" in msg

        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False
    finally:
        if log_path.exists():
            log_path.unlink()


def test_link():
    """Test linking events"""
    print("Testing LINK...", end=" ")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        log_path = Path(f.name)

    try:
        state = ChronosState(log_path)

        # Start task
        state.start("test_task")

        # Get event ID
        events = state.event_log.read_all()
        event_id = events[0].id

        # Link to external entity
        msg = state.link(event_id, "external_task_123", relation="part_of")
        assert "Linked" in msg
        assert "part_of" in msg

        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False
    finally:
        if log_path.exists():
            log_path.unlink()


def test_persistence():
    """Test state persistence across instances"""
    print("Testing persistence...", end=" ")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        log_path = Path(f.name)

    try:
        # First instance
        state1 = ChronosState(log_path)
        state1.start("persistent_task")

        # Second instance
        state2 = ChronosState(log_path)
        assert state2.active_task is not None
        assert state2.active_task.task == "persistent_task"

        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False
    finally:
        if log_path.exists():
            log_path.unlink()


def test_voice_parser():
    """Test voice command parsing"""
    print("Testing voice parser...", end=" ")

    try:
        from chronos.voice import VoiceCommandParser

        parser = VoiceCommandParser()

        # Test start
        cmd = parser.parse("start compost sifting")
        assert cmd["verb"] == "start"
        assert cmd["args"]["task"] == "compost_sifting"

        # Test stop
        cmd = parser.parse("done")
        assert cmd["verb"] == "stop"

        # Test note
        cmd = parser.parse("note: test note")
        assert cmd["verb"] == "note"

        # Test with tags
        cmd = parser.parse("start test task #tag1 #tag2")
        assert "tag1" in cmd["args"]["tags"]
        assert "tag2" in cmd["args"]["tags"]

        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_event_log():
    """Test basic event log operations"""
    print("Testing event log...", end=" ")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        log_path = Path(f.name)

    try:
        log = EventLog(log_path)

        # Append events
        e1 = log.append(ChronosEvent(event="start", task="t1", ts="", id=""))
        e2 = log.append(ChronosEvent(event="stop", task="t1", ts="", id=""))

        # Check IDs are sequential
        assert e1.id == "chr_0001"
        assert e2.id == "chr_0002"

        # Read back
        events = log.read_all()
        assert len(events) == 2

        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False
    finally:
        if log_path.exists():
            log_path.unlink()


def main():
    """Run all tests"""
    print("=" * 50)
    print("Chronos Core Tests")
    print("=" * 50)
    print()

    tests = [
        test_event_log,
        test_start_stop,
        test_log_retroactive,
        test_note,
        test_link,
        test_persistence,
        test_voice_parser,
    ]

    results = []
    for test in tests:
        results.append(test())

    print()
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
