"""
Chronos Core Tests - Golden path validation

Tests the 5 verbs: start, stop, log, note, link
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from chronos.core import ChronosState, EventLog, ChronosEvent


@pytest.fixture
def temp_log():
    """Create temporary log file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        log_path = Path(f.name)

    yield log_path

    # Cleanup
    if log_path.exists():
        log_path.unlink()


class TestEventLog:
    """Test append-only event log"""

    def test_create_log(self, temp_log):
        """Test creating a new log file"""
        log = EventLog(temp_log)
        assert temp_log.exists()

    def test_append_event(self, temp_log):
        """Test appending events"""
        log = EventLog(temp_log)

        event = ChronosEvent(
            event="start",
            task="test_task",
            ts="",
            id=""
        )

        result = log.append(event)

        # Should generate ID and timestamp
        assert result.id.startswith("chr_")
        assert result.ts  # ISO timestamp

    def test_read_events(self, temp_log):
        """Test reading events from log"""
        log = EventLog(temp_log)

        # Append some events
        log.append(ChronosEvent(event="start", task="task1", ts="", id=""))
        log.append(ChronosEvent(event="stop", task="task1", ts="", id=""))

        # Read back
        events = log.read_all()
        assert len(events) == 2
        assert events[0].event == "start"
        assert events[1].event == "stop"

    def test_event_id_sequence(self, temp_log):
        """Test event IDs are sequential"""
        log = EventLog(temp_log)

        e1 = log.append(ChronosEvent(event="start", task="t1", ts="", id=""))
        e2 = log.append(ChronosEvent(event="stop", task="t1", ts="", id=""))

        assert e1.id == "chr_0001"
        assert e2.id == "chr_0002"


class TestChronosState:
    """Test the 5 verbs"""

    def test_start_task(self, temp_log):
        """Test START verb"""
        state = ChronosState(temp_log)
        msg = state.start("compost_sifting", tags=["farm", "manual"])

        assert "Started compost_sifting" in msg
        assert "#farm" in msg
        assert "#manual" in msg
        assert state.active_task is not None
        assert state.active_task.task == "compost_sifting"

    def test_stop_task(self, temp_log):
        """Test STOP verb"""
        state = ChronosState(temp_log)

        # Start a task
        state.start("test_task")

        # Stop it
        msg = state.stop()

        assert "Stopped test_task" in msg
        assert "duration" in msg
        assert state.active_task is None

    def test_stop_without_active_task(self, temp_log):
        """Test STOP when no active task"""
        state = ChronosState(temp_log)
        msg = state.stop()

        assert "No active task" in msg

    def test_log_completed_task(self, temp_log):
        """Test LOG verb (retroactive entry)"""
        state = ChronosState(temp_log)

        msg = state.log("email_review", duration_sec=1200, tags=["admin"])

        assert "Logged email_review" in msg
        assert "20m" in msg
        assert "#admin" in msg

    def test_note_on_active_task(self, temp_log):
        """Test NOTE verb"""
        state = ChronosState(temp_log)

        # Start a task
        state.start("compost_sifting")

        # Add note
        msg = state.note("Compost ready for spring beds")

        assert "Note added" in msg
        assert "compost_sifting" in msg

    def test_note_without_active_task(self, temp_log):
        """Test NOTE without active task fails"""
        state = ChronosState(temp_log)
        msg = state.note("Random note")

        assert "No active task" in msg

    def test_note_with_explicit_link(self, temp_log):
        """Test NOTE with explicit event ID"""
        state = ChronosState(temp_log)

        # Create an event
        state.start("test_task")
        events = state.log.read_all()
        event_id = events[0].id

        state.stop()

        # Add note to completed task
        msg = state.note("Retrospective note", linked_to=event_id)

        assert "Note added" in msg

    def test_link_events(self, temp_log):
        """Test LINK verb"""
        state = ChronosState(temp_log)

        # Create an event
        state.start("garden_task")
        events = state.log.read_all()
        event_id = events[0].id

        # Link to external entity
        msg = state.link(event_id, "task_garden_prep_2025", relation="part_of")

        assert "Linked" in msg
        assert "part_of" in msg

    def test_status_with_active_task(self, temp_log):
        """Test STATUS with active task"""
        state = ChronosState(temp_log)
        state.start("test_task")

        msg = state.status()

        assert "Active" in msg
        assert "test_task" in msg

    def test_status_without_active_task(self, temp_log):
        """Test STATUS without active task"""
        state = ChronosState(temp_log)
        msg = state.status()

        assert "No active task" in msg

    def test_auto_stop_on_new_start(self, temp_log):
        """Test that starting new task auto-stops previous"""
        state = ChronosState(temp_log)

        # Start first task
        state.start("task1")

        # Start second task (should auto-stop first)
        state.start("task2")

        # Check that task2 is now active
        assert state.active_task.task == "task2"

        # Check that task1 was stopped
        events = state.log.read_all()
        stop_events = [e for e in events if e.event == "stop" and e.task == "task1"]
        assert len(stop_events) == 1

    def test_persistence_across_instances(self, temp_log):
        """Test that state persists across ChronosState instances"""
        # First instance: start a task
        state1 = ChronosState(temp_log)
        state1.start("persistent_task")

        # Second instance: should reload active task
        state2 = ChronosState(temp_log)
        assert state2.active_task is not None
        assert state2.active_task.task == "persistent_task"

        # Stop in second instance
        state2.stop()

        # Third instance: no active task
        state3 = ChronosState(temp_log)
        assert state3.active_task is None


class TestVoiceCommands:
    """Test voice command parsing"""

    def test_parse_start_command(self):
        """Test parsing START commands"""
        from chronos.voice import VoiceCommandParser

        parser = VoiceCommandParser()

        cmd = parser.parse("start compost sifting")
        assert cmd["verb"] == "start"
        assert cmd["args"]["task"] == "compost_sifting"

        cmd = parser.parse("begin email review")
        assert cmd["verb"] == "start"
        assert cmd["args"]["task"] == "email_review"

    def test_parse_stop_command(self):
        """Test parsing STOP commands"""
        from chronos.voice import VoiceCommandParser

        parser = VoiceCommandParser()

        for phrase in ["stop", "done", "finished", "complete"]:
            cmd = parser.parse(phrase)
            assert cmd["verb"] == "stop"

    def test_parse_note_command(self):
        """Test parsing NOTE commands"""
        from chronos.voice import VoiceCommandParser

        parser = VoiceCommandParser()

        cmd = parser.parse("note: compost ready for spring")
        assert cmd["verb"] == "note"
        assert "compost ready" in cmd["args"]["text"]

        cmd = parser.parse("remember to check moisture")
        assert cmd["verb"] == "note"
        assert "check moisture" in cmd["args"]["text"]

    def test_parse_log_command(self):
        """Test parsing LOG commands"""
        from chronos.voice import VoiceCommandParser

        parser = VoiceCommandParser()

        cmd = parser.parse("log email review for 20 minutes")
        assert cmd["verb"] == "log"
        assert cmd["args"]["task"] == "email_review"
        assert cmd["args"]["duration_sec"] == 1200  # 20 minutes

    def test_parse_with_tags(self):
        """Test parsing commands with hashtags"""
        from chronos.voice import VoiceCommandParser

        parser = VoiceCommandParser()

        cmd = parser.parse("start compost sifting #farm #manual")
        assert cmd["verb"] == "start"
        assert cmd["args"]["task"] == "compost_sifting"
        assert "farm" in cmd["args"]["tags"]
        assert "manual" in cmd["args"]["tags"]

    def test_parse_invalid_command(self):
        """Test that invalid commands return None"""
        from chronos.voice import VoiceCommandParser

        parser = VoiceCommandParser()

        cmd = parser.parse("random gibberish that makes no sense")
        assert cmd is None

    def test_format_confirmation(self):
        """Test confirmation message formatting"""
        from chronos.voice import format_confirmation

        cmd = {"verb": "start", "args": {"task": "compost_sifting"}}
        msg = format_confirmation(cmd)
        assert "Starting compost sifting" in msg

        cmd = {"verb": "stop", "args": {}}
        msg = format_confirmation(cmd)
        assert "Stopping" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
