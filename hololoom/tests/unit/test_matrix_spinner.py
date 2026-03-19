"""
Tests for MatrixSpinner

Tests matrix-nio integration, event parsing, and MemoryShard conversion.
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

# Mock matrix-nio before importing MatrixSpinner
class MockRoomMessage:
    def __init__(self, event_id, sender, body, server_timestamp, msgtype="m.text"):
        self.event_id = event_id
        self.sender = sender
        self.body = body
        self.server_timestamp = server_timestamp
        self.msgtype = msgtype
        self.source = {'content': {}}

class MockRoomMessageText(MockRoomMessage):
    def __init__(self, event_id, sender, body, server_timestamp):
        super().__init__(event_id, sender, body, server_timestamp, "m.text")
        self.formatted_body = None

class MockRoomMessageMedia(MockRoomMessage):
    def __init__(self, event_id, sender, body, server_timestamp, url, msgtype="m.image"):
        super().__init__(event_id, sender, body, server_timestamp, msgtype)
        self.url = url

class MockMatrixRoom:
    def __init__(self, room_id, display_name):
        self.room_id = room_id
        self.display_name = display_name

    def user_name(self, user_id):
        return user_id.split(':')[0][1:]  # @user:server -> user

class MockAsyncClient:
    def __init__(self, homeserver, user_id, device_id=None):
        self.homeserver = homeserver
        self.user_id = user_id
        self.device_id = device_id
        self.access_token = "mock_token"
        self.next_batch = "mock_batch_token"
        self.rooms = {}

    async def login(self, password):
        return Mock(status_code=200)

    async def sync(self, timeout=30000, full_state=False):
        return Mock(status_code=200)

    async def room_messages(self, room_id, start, limit, direction="b"):
        # Return mock response
        return Mock(
            chunk=[],
            status_code=200
        )

    async def close(self):
        pass

# Patch matrix-nio imports
sys_modules_patch = {
    'nio': MagicMock(
        AsyncClient=MockAsyncClient,
        RoomMessageText=MockRoomMessageText,
        RoomMessageMedia=MockRoomMessageMedia,
        RoomMessage=MockRoomMessage,
        MatrixRoom=MockMatrixRoom,
        LoginError=Exception,
        SyncError=Exception,
        RoomMessagesError=Exception
    )
}

with patch.dict('sys.modules', sys_modules_patch):
    from hololoom.spinningWheel.matrix_spinner import (
        MatrixSpinner,
        MatrixParser,
        MatrixMessage,
        spin_matrix_room,
        create_matrix_scorer
    )
    from hololoom.spinningWheel.protocol import SpinnerCheckpoint

# Patch module-level globals in matrix_spinner so parse_event works correctly.
# The spinningWheel __init__.py imports matrix_spinner eagerly before the test
# patches sys.modules, so MATRIX_AVAILABLE is False and RoomMessage/etc are
# not bound.  We fix that here using sys.modules to avoid re-triggering the
# full spinningWheel import chain.
import sys as _sys
_mx_mod = _sys.modules.get('hololoom.spinningWheel.matrix_spinner')
if _mx_mod is not None:
    _mx_mod.MATRIX_AVAILABLE = True
    _mx_mod.RoomMessage = MockRoomMessage
    _mx_mod.RoomMessageText = MockRoomMessageText
    _mx_mod.RoomMessageMedia = MockRoomMessageMedia


# Fixtures

@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_matrix_room():
    """Create mock Matrix room"""
    return MockMatrixRoom(
        room_id="!test:matrix.org",
        display_name="Test Room"
    )


@pytest.fixture
def mock_text_event():
    """Create mock text message event"""
    event = MockRoomMessageText(
        event_id="$event1",
        sender="@alice:matrix.org",
        body="This is a test message about Thompson Sampling",
        server_timestamp=int(datetime.now().timestamp() * 1000)
    )
    return event


@pytest.fixture
def mock_image_event():
    """Create mock image message event"""
    event = MockRoomMessageMedia(
        event_id="$event2",
        sender="@bob:matrix.org",
        body="diagram.png",
        server_timestamp=int(datetime.now().timestamp() * 1000),
        url="mxc://matrix.org/abc123",
        msgtype="m.image"
    )
    return event


@pytest.fixture
def mock_reply_event():
    """Create mock reply event"""
    event = MockRoomMessageText(
        event_id="$event3",
        sender="@charlie:matrix.org",
        body="Great point about @alice Thompson Sampling!",
        server_timestamp=int(datetime.now().timestamp() * 1000)
    )
    event.source = {
        'content': {
            'm.relates_to': {
                'm.in_reply_to': {
                    'event_id': '$event1'
                }
            }
        }
    }
    return event


# MatrixParser Tests

def test_matrix_parser_text_event(mock_text_event, mock_matrix_room):
    """Test parsing text message event"""
    msg = MatrixParser.parse_event(mock_text_event, mock_matrix_room)

    assert msg is not None
    assert msg.event_id == "$event1"
    assert msg.room_id == "!test:matrix.org"
    assert msg.room_name == "Test Room"
    assert msg.sender == "@alice:matrix.org"
    assert msg.msg_type == "m.text"
    assert "Thompson Sampling" in msg.content
    assert msg.is_thread_root  # Not a reply


def test_matrix_parser_image_event(mock_image_event, mock_matrix_room):
    """Test parsing image message event"""
    msg = MatrixParser.parse_event(mock_image_event, mock_matrix_room)

    assert msg is not None
    assert msg.event_id == "$event2"
    assert msg.msg_type == "m.image"
    assert msg.content == "diagram.png"
    assert "mxc://matrix.org/abc123" in msg.attachments


def test_matrix_parser_reply_event(mock_reply_event, mock_matrix_room):
    """Test parsing reply event"""
    msg = MatrixParser.parse_event(mock_reply_event, mock_matrix_room)

    assert msg is not None
    assert msg.reply_to == "$event1"
    assert not msg.is_thread_root  # This is a reply
    assert "alice" in msg.mentions


def test_matrix_parser_extract_mentions():
    """Test mention extraction"""
    text = "Hey @alice and @bob:matrix.org, check this out!"
    mentions = MatrixParser.extract_mentions(text)

    assert "alice" in mentions
    assert "bob:matrix.org" in mentions
    assert len(mentions) == 2


# MatrixSpinner Tests

@pytest.mark.asyncio
async def test_matrix_spinner_initialization():
    """Test MatrixSpinner initialization"""
    spinner = MatrixSpinner(
        homeserver="https://matrix.org",
        access_token="test_token",
        importance_threshold=0.3
    )

    assert spinner.get_name() == "matrix"
    assert spinner.homeserver == "https://matrix.org"
    assert spinner.access_token == "test_token"
    assert spinner.importance_threshold == 0.3


@pytest.mark.asyncio
async def test_matrix_spinner_capabilities():
    """Test spinner capabilities"""
    spinner = MatrixSpinner(
        homeserver="https://matrix.org",
        access_token="test_token"
    )

    caps = spinner.get_capabilities()
    assert caps.basic_processing
    assert caps.entity_extraction
    assert caps.motif_extraction
    assert caps.importance_scoring
    assert caps.incremental
    assert caps.streaming
    assert 'matrix' in caps.supported_formats


@pytest.mark.asyncio
async def test_matrix_spinner_availability():
    """Test spinner availability check"""
    spinner = MatrixSpinner(
        homeserver="https://matrix.org",
        access_token="test_token"
    )

    # Should be available since we mocked nio
    assert spinner.is_available()


@pytest.mark.asyncio
async def test_matrix_spinner_format_message():
    """Test message formatting"""
    spinner = MatrixSpinner(
        homeserver="https://matrix.org",
        access_token="test_token"
    )

    msg = MatrixMessage(
        event_id="$event1",
        room_id="!room:matrix.org",
        room_name="Test Room",
        sender="@alice:matrix.org",
        sender_display_name="Alice",
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
        msg_type="m.text",
        content="Hello world",
        reactions={"👍": 3, "🎉": 1},
        mentions=["bob"]
    )

    formatted = spinner._format_message_text(msg)

    assert "Alice" in formatted
    assert "Hello world" in formatted
    assert "👍:3" in formatted
    assert "bob" in formatted


@pytest.mark.asyncio
async def test_matrix_spinner_extract_entities():
    """Test entity extraction"""
    spinner = MatrixSpinner(
        homeserver="https://matrix.org",
        access_token="test_token"
    )

    msg = MatrixMessage(
        event_id="$event1",
        room_id="!room:matrix.org",
        room_name="Development",
        sender="@alice:matrix.org",
        sender_display_name="Alice",
        timestamp=datetime.now(),
        msg_type="m.text",
        content="Hey @bob",
        mentions=["bob", "charlie"]
    )

    entities = spinner._extract_entities(msg)

    assert "Alice" in entities
    assert "Development" in entities
    assert "bob" in entities
    assert "charlie" in entities


@pytest.mark.asyncio
async def test_matrix_spinner_extract_motifs():
    """Test motif extraction"""
    spinner = MatrixSpinner(
        homeserver="https://matrix.org",
        access_token="test_token"
    )

    # Thread root with reactions and mentions
    msg = MatrixMessage(
        event_id="$event1",
        room_id="!room:matrix.org",
        room_name="Test",
        sender="@alice:matrix.org",
        sender_display_name="Alice",
        timestamp=datetime.now(),
        msg_type="m.text",
        content="Check this out @bob",
        reactions={"👍": 5},
        mentions=["bob"],
        attachments=["mxc://matrix.org/file"]
    )

    motifs = spinner._extract_motifs(msg)

    assert "m.text" in motifs
    assert "thread_root" in motifs
    assert "reacted" in motifs
    assert "mentions" in motifs
    assert "media" in motifs


@pytest.mark.asyncio
async def test_matrix_spinner_importance_scoring():
    """Test importance scoring"""
    spinner = MatrixSpinner(
        homeserver="https://matrix.org",
        access_token="test_token"
    )

    # High importance message
    msg_important = MatrixMessage(
        event_id="$event1",
        room_id="!room:matrix.org",
        room_name="Development",
        sender="@alice:matrix.org",
        sender_display_name="Alice",
        timestamp=datetime.now(),
        msg_type="m.text",
        content="We need to fix the critical bug in the API endpoint. This affects production deployments and needs immediate attention.",
        reactions={"👍": 5, "🔥": 2},
        mentions=["bob", "charlie"]
    )

    score_important = spinner.score_importance(msg_important)
    assert score_important.score > 0.5

    # Low importance message
    msg_noise = MatrixMessage(
        event_id="$event2",
        room_id="!room:matrix.org",
        room_name="Random",
        sender="@bob:matrix.org",
        sender_display_name="Bob",
        timestamp=datetime.now(),
        msg_type="m.text",
        content="lol"
    )

    score_noise = spinner.score_importance(msg_noise)
    assert score_noise.score < 0.5  # Low importance for noise


@pytest.mark.asyncio
async def test_matrix_spinner_engagement_score():
    """Test engagement score calculation"""
    # High engagement
    msg_high = MatrixMessage(
        event_id="$event1",
        room_id="!room:matrix.org",
        room_name="Test",
        sender="@alice:matrix.org",
        sender_display_name="Alice",
        timestamp=datetime.now(),
        msg_type="m.text",
        content="Important announcement",
        reactions={"👍": 10, "🎉": 5, "❤️": 3}
    )

    assert msg_high.engagement_score >= 1.0  # Capped at 1.0

    # No engagement
    msg_low = MatrixMessage(
        event_id="$event2",
        room_id="!room:matrix.org",
        room_name="Test",
        sender="@bob:matrix.org",
        sender_display_name="Bob",
        timestamp=datetime.now(),
        msg_type="m.text",
        content="Hello"
    )

    assert msg_low.engagement_score == 0.0


@pytest.mark.asyncio
async def test_matrix_spinner_messages_to_shards():
    """Test converting messages to shards"""
    spinner = MatrixSpinner(
        homeserver="https://matrix.org",
        access_token="test_token",
        importance_threshold=0.3
    )

    messages = [
        MatrixMessage(
            event_id="$event1",
            room_id="!room:matrix.org",
            room_name="Development",
            sender="@alice:matrix.org",
            sender_display_name="Alice",
            timestamp=datetime.now(),
            msg_type="m.text",
            content="We should implement feature X using algorithm Y. This will improve performance significantly.",
            reactions={"👍": 3}
        ),
        MatrixMessage(
            event_id="$event2",
            room_id="!room:matrix.org",
            room_name="Random",
            sender="@bob:matrix.org",
            sender_display_name="Bob",
            timestamp=datetime.now(),
            msg_type="m.text",
            content="hi"  # Should be filtered (noise)
        )
    ]

    shards = spinner._messages_to_shards(messages)

    # First message should pass (high importance)
    # Second message should be filtered (noise)
    assert len(shards) >= 1
    assert shards[0].metadata['event_id'] == "$event1"
    assert shards[0].metadata['importance_score'] >= 0.3


@pytest.mark.asyncio
async def test_matrix_spinner_create_source_id():
    """Test source ID generation"""
    spinner = MatrixSpinner(
        homeserver="https://matrix.org",
        access_token="test_token"
    )

    source_id_1 = spinner._create_source_id("!room1:matrix.org")
    source_id_2 = spinner._create_source_id("!room2:matrix.org")
    source_id_3 = spinner._create_source_id("!room1:matrix.org")  # Same as 1

    # Should be deterministic
    assert source_id_1 == source_id_3

    # Should be different for different rooms
    assert source_id_1 != source_id_2

    # Should be short
    assert len(source_id_1) == 16


@pytest.mark.asyncio
async def test_create_matrix_scorer():
    """Test Matrix-specific importance scorer creation"""
    scorer = create_matrix_scorer()

    # Should have technical terms
    assert len(scorer.technical_scorer.technical_terms) > 0

    # Test scoring
    technical_text = "We need to fix the bug in the API endpoint and deploy to production."
    noise_text = "lol"

    tech_score = scorer.technical_scorer.score(technical_text)
    noise_score = scorer.noise_detector.detect(noise_text)

    assert tech_score > 0.0  # Should detect technical terms
    assert noise_score < 0.0  # Should detect noise (returns negative penalty)


@pytest.mark.asyncio
async def test_matrix_message_is_thread_root():
    """Test thread root detection"""
    # Thread root (no reply_to, no thread_root)
    msg1 = MatrixMessage(
        event_id="$event1",
        room_id="!room:matrix.org",
        room_name="Test",
        sender="@alice:matrix.org",
        sender_display_name="Alice",
        timestamp=datetime.now(),
        msg_type="m.text",
        content="Original message"
    )

    assert msg1.is_thread_root

    # Reply (has reply_to)
    msg2 = MatrixMessage(
        event_id="$event2",
        room_id="!room:matrix.org",
        room_name="Test",
        sender="@bob:matrix.org",
        sender_display_name="Bob",
        timestamp=datetime.now(),
        msg_type="m.text",
        content="Reply",
        reply_to="$event1"
    )

    assert not msg2.is_thread_root


@pytest.mark.asyncio
async def test_matrix_spinner_close():
    """Test spinner cleanup"""
    spinner = MatrixSpinner(
        homeserver="https://matrix.org",
        access_token="test_token"
    )

    # Should not raise
    await spinner.close()

    # Should handle double close
    await spinner.close()


# Note: Integration tests with real matrix-nio would go here
# For now, we test the spinner functionality with mocked events


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
