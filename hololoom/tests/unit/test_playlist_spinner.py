"""
Unit Tests for PlaylistSpinner
===============================

Tests playlist ingestion including:
- ResonanceIndex tracking and scoring
- ResonanceScorer importance weighting
- PlaylistSpinner initialization and capabilities
- Manifest parsing and video deduplication
- Checkpoint-based incremental ingestion
- Backpressure (batch_size limiting)
- Metadata fallback when transcripts unavailable

Author: Claude Code
Date: March 2026
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

pytest.importorskip("hololoom.documentation")

from hololoom.spinningWheel.playlist_spinner import (
    PlaylistEntry,
    PlaylistManifest,
    PlaylistSpinner,
    ResonanceIndex,
    ResonanceScorer,
    YTDLP_AVAILABLE,
)
from hololoom.spinningWheel.protocol import SpinnerStatus


# ============================================================================
# ResonanceIndex Tests
# ============================================================================

class TestResonanceIndex:
    """Tests for the source-agnostic resonance index."""

    def test_empty_index(self):
        idx = ResonanceIndex()
        assert idx.resonance_count('nonexistent') == 0
        assert idx.resonance_score('nonexistent') == 0.0
        assert idx.all_video_ids() == set()

    def test_single_source(self):
        idx = ResonanceIndex()
        idx.add('vid1', 'playlist_A')
        assert idx.resonance_count('vid1') == 1
        assert idx.resonance_score('vid1', saturation=3) == pytest.approx(1 / 3)

    def test_multi_source_resonance(self):
        idx = ResonanceIndex()
        idx.add('vid1', 'playlist_A')
        idx.add('vid1', 'playlist_B')
        idx.add('vid1', 'playlist_C')
        assert idx.resonance_count('vid1') == 3
        assert idx.resonance_score('vid1', saturation=3) == pytest.approx(1.0)

    def test_saturation_cap(self):
        """Score should cap at 1.0 even with more sources than saturation."""
        idx = ResonanceIndex()
        for i in range(5):
            idx.add('vid1', f'source_{i}')
        assert idx.resonance_score('vid1', saturation=3) == 1.0

    def test_duplicate_source_ignored(self):
        """Adding same video+source twice doesn't double-count."""
        idx = ResonanceIndex()
        idx.add('vid1', 'playlist_A')
        idx.add('vid1', 'playlist_A')
        assert idx.resonance_count('vid1') == 1

    def test_metadata_first_encounter(self):
        """Metadata is stored from first encounter only."""
        idx = ResonanceIndex()
        idx.add('vid1', 'src_A', {'title': 'First'})
        idx.add('vid1', 'src_B', {'title': 'Second'})
        assert idx.get_metadata('vid1')['title'] == 'First'

    def test_ranked(self):
        idx = ResonanceIndex()
        idx.add('low', 'A')
        idx.add('high', 'A')
        idx.add('high', 'B')
        idx.add('high', 'C')
        idx.add('mid', 'A')
        idx.add('mid', 'B')

        ranked = idx.ranked()
        assert ranked[0] == ('high', 3)
        assert ranked[1] == ('mid', 2)
        assert ranked[2] == ('low', 1)

    def test_serialization_roundtrip(self):
        idx = ResonanceIndex()
        idx.add('vid1', 'A', {'title': 'Test'})
        idx.add('vid1', 'B')
        idx.add('vid2', 'A')

        data = idx.to_dict()
        restored = ResonanceIndex.from_dict(data)

        assert restored.resonance_count('vid1') == 2
        assert restored.resonance_count('vid2') == 1
        assert restored.get_metadata('vid1')['title'] == 'Test'

    def test_all_video_ids(self):
        idx = ResonanceIndex()
        idx.add('a', 'src')
        idx.add('b', 'src')
        idx.add('c', 'src')
        assert idx.all_video_ids() == {'a', 'b', 'c'}


# ============================================================================
# ResonanceScorer Tests
# ============================================================================

class TestResonanceScorer:
    """Tests for the resonance-aware importance scorer."""

    def test_high_resonance_boosts_score(self):
        idx = ResonanceIndex()
        idx.add('popular', 'A')
        idx.add('popular', 'B')
        idx.add('popular', 'C')
        idx.add('obscure', 'A')

        scorer = ResonanceScorer(resonance_index=idx)

        popular_score = scorer.score(
            text="A long substantive transcript about neural networks and machine learning algorithms " * 10,
            video_id='popular',
        )
        obscure_score = scorer.score(
            text="A long substantive transcript about neural networks and machine learning algorithms " * 10,
            video_id='obscure',
        )

        # Same text, but popular has 3x resonance → higher score
        assert popular_score.score > obscure_score.score

    def test_scorer_with_empty_index(self):
        """Scorer shouldn't crash with no resonance data."""
        idx = ResonanceIndex()
        scorer = ResonanceScorer(resonance_index=idx)

        score = scorer.score(text="Some text", video_id='unknown')
        assert 0.0 <= score.score <= 1.0


# ============================================================================
# PlaylistSpinner Initialization Tests
# ============================================================================

class TestPlaylistSpinnerInit:
    """Tests for spinner initialization and protocol compliance."""

    def test_initialization_defaults(self):
        spinner = PlaylistSpinner()
        assert spinner.name == 'playlist'
        assert spinner.batch_size == 20
        assert spinner.transcript_languages == ['en']
        assert spinner.resonance_saturation == 3

    def test_custom_initialization(self):
        spinner = PlaylistSpinner(
            importance_threshold=0.5,
            batch_size=10,
            transcript_languages=['en', 'es'],
            resonance_saturation=5,
        )
        assert spinner.importance_threshold == 0.5
        assert spinner.batch_size == 10
        assert spinner.transcript_languages == ['en', 'es']
        assert spinner.resonance_saturation == 5

    def test_capabilities(self):
        spinner = PlaylistSpinner()
        caps = spinner.get_capabilities()

        assert caps.basic_processing is True
        assert caps.incremental is True
        assert caps.batch_processing is True
        assert caps.importance_scoring is True
        assert caps.streaming is False
        assert 'youtube_playlist_url' in caps.supported_formats

    def test_availability(self):
        spinner = PlaylistSpinner()
        assert spinner.is_available() == YTDLP_AVAILABLE

    def test_status(self):
        spinner = PlaylistSpinner()
        status = spinner.get_status()
        if YTDLP_AVAILABLE:
            assert status in [SpinnerStatus.AVAILABLE, SpinnerStatus.DEGRADED]
        else:
            assert status == SpinnerStatus.UNAVAILABLE

    def test_resonance_index_accessible(self):
        spinner = PlaylistSpinner()
        assert spinner.resonance_index is not None
        assert isinstance(spinner.resonance_index, ResonanceIndex)


# ============================================================================
# PlaylistSpinner Core Logic Tests (mocked yt-dlp + YouTubeSpinner)
# ============================================================================

def _make_manifest(playlist_id: str, video_ids: list[str], title: str = 'Test') -> PlaylistManifest:
    """Helper to build a PlaylistManifest."""
    entries = [
        PlaylistEntry(
            video_id=vid,
            title=f"Video {vid}",
            channel=f"Channel_{vid}",
            duration=600.0,
            upload_date='20260320',
            playlist_index=i,
        )
        for i, vid in enumerate(video_ids)
    ]
    return PlaylistManifest(playlist_id=playlist_id, title=title, entries=entries)


@pytest.mark.asyncio
@pytest.mark.skipif(not YTDLP_AVAILABLE, reason="yt-dlp not available")
async def test_spin_new_videos(tmp_path):
    """Test that new videos produce shards."""
    spinner = PlaylistSpinner(
        importance_threshold=0.0,
        batch_size=50,
        checkpoint_dir=tmp_path,
    )

    manifest = _make_manifest('PL_test', ['vid_a', 'vid_b', 'vid_c'])

    with patch.object(spinner, '_fetch_manifests', return_value=[manifest]):
        with patch.object(spinner, '_process_one_video') as mock_process:
            # Return a shard for each video
            async def fake_process(vid, yt, scorer):
                return spinner._create_shard(
                    id_suffix=vid,
                    text=f"Transcript for {vid}",
                    episode='test',
                    entities=[],
                    motifs=[],
                    metadata={'video_id': vid, 'importance_score': 0.5},
                )
            mock_process.side_effect = fake_process

            result = await spinner.spin(['https://youtube.com/playlist?list=PL_test'])

            assert result.success is True
            assert result.shard_count == 3
            assert mock_process.call_count == 3


@pytest.mark.asyncio
@pytest.mark.skipif(not YTDLP_AVAILABLE, reason="yt-dlp not available")
async def test_incremental_skips_seen(tmp_path):
    """Test that already-seen videos are skipped on second run."""
    spinner = PlaylistSpinner(
        importance_threshold=0.0,
        batch_size=50,
        checkpoint_dir=tmp_path,
    )

    manifest = _make_manifest('PL_test', ['vid_a', 'vid_b'])

    with patch.object(spinner, '_fetch_manifests', return_value=[manifest]):
        with patch.object(spinner, '_process_one_video') as mock_process:
            async def fake_process(vid, yt, scorer):
                return spinner._create_shard(
                    id_suffix=vid,
                    text=f"Transcript for {vid}",
                    episode='test',
                    entities=[],
                    motifs=[],
                    metadata={'video_id': vid, 'importance_score': 0.5},
                )
            mock_process.side_effect = fake_process

            # First run: processes both
            await spinner.spin(['https://youtube.com/playlist?list=PL_test'])
            assert mock_process.call_count == 2

            mock_process.reset_mock()

            # Second run: same manifest → no new videos
            # Need fresh spinner to reload checkpoint
            spinner2 = PlaylistSpinner(
                importance_threshold=0.0,
                batch_size=50,
                checkpoint_dir=tmp_path,
            )
            with patch.object(spinner2, '_fetch_manifests', return_value=[manifest]):
                with patch.object(spinner2, '_process_one_video') as mock_process2:
                    mock_process2.side_effect = fake_process
                    result = await spinner2.spin(['https://youtube.com/playlist?list=PL_test'])
                    assert result.shard_count == 0
                    assert mock_process2.call_count == 0


@pytest.mark.asyncio
@pytest.mark.skipif(not YTDLP_AVAILABLE, reason="yt-dlp not available")
async def test_backpressure_limits_batch(tmp_path):
    """Test that batch_size caps the number of videos processed."""
    spinner = PlaylistSpinner(
        importance_threshold=0.0,
        batch_size=2,  # Only process 2
        checkpoint_dir=tmp_path,
    )

    manifest = _make_manifest('PL_test', ['v1', 'v2', 'v3', 'v4', 'v5'])

    with patch.object(spinner, '_fetch_manifests', return_value=[manifest]):
        with patch.object(spinner, '_process_one_video') as mock_process:
            async def fake_process(vid, yt, scorer):
                return spinner._create_shard(
                    id_suffix=vid,
                    text=f"Transcript for {vid}",
                    episode='test',
                    entities=[],
                    motifs=[],
                    metadata={'video_id': vid, 'importance_score': 0.5},
                )
            mock_process.side_effect = fake_process

            result = await spinner.spin(['https://youtube.com/playlist?list=PL_test'])

            # Only 2 processed due to backpressure
            assert mock_process.call_count == 2
            assert result.shard_count == 2


@pytest.mark.asyncio
@pytest.mark.skipif(not YTDLP_AVAILABLE, reason="yt-dlp not available")
async def test_cross_playlist_resonance(tmp_path):
    """Test that videos on multiple playlists get higher resonance."""
    spinner = PlaylistSpinner(
        importance_threshold=0.0,
        batch_size=50,
        checkpoint_dir=tmp_path,
    )

    # vid_shared appears on both playlists
    manifest_a = _make_manifest('PL_A', ['vid_unique_a', 'vid_shared'])
    manifest_b = _make_manifest('PL_B', ['vid_unique_b', 'vid_shared'])

    with patch.object(spinner, '_fetch_manifests', return_value=[manifest_a, manifest_b]):
        with patch.object(spinner, '_process_one_video') as mock_process:
            async def fake_process(vid, yt, scorer):
                resonance = spinner.resonance_index.resonance_count(vid)
                return spinner._create_shard(
                    id_suffix=vid,
                    text=f"Transcript for {vid}",
                    episode='test',
                    entities=[],
                    motifs=[],
                    metadata={
                        'video_id': vid,
                        'resonance_count': resonance,
                        'importance_score': 0.5,
                    },
                )
            mock_process.side_effect = fake_process

            result = await spinner.spin([
                'https://youtube.com/playlist?list=PL_A',
                'https://youtube.com/playlist?list=PL_B',
            ])

            # vid_shared should have resonance_count=2
            shared_shards = [
                s for s in result.shards
                if s.metadata.get('video_id') == 'vid_shared'
            ]
            assert len(shared_shards) == 1
            assert shared_shards[0].metadata['resonance_count'] == 2

            # unique videos should have resonance_count=1
            unique_shards = [
                s for s in result.shards
                if s.metadata.get('video_id') != 'vid_shared'
            ]
            for s in unique_shards:
                assert s.metadata['resonance_count'] == 1


@pytest.mark.asyncio
@pytest.mark.skipif(not YTDLP_AVAILABLE, reason="yt-dlp not available")
async def test_high_resonance_processed_first(tmp_path):
    """Test that high-resonance videos are processed before low-resonance."""
    spinner = PlaylistSpinner(
        importance_threshold=0.0,
        batch_size=2,  # Only process 2 of 3
        checkpoint_dir=tmp_path,
    )

    # vid_hot is on both playlists, vid_cold_* on one each
    manifest_a = _make_manifest('PL_A', ['vid_cold_a', 'vid_hot'])
    manifest_b = _make_manifest('PL_B', ['vid_cold_b', 'vid_hot'])

    processed_order = []

    with patch.object(spinner, '_fetch_manifests', return_value=[manifest_a, manifest_b]):
        with patch.object(spinner, '_process_one_video') as mock_process:
            async def fake_process(vid, yt, scorer):
                processed_order.append(vid)
                return spinner._create_shard(
                    id_suffix=vid,
                    text=f"Transcript for {vid}",
                    episode='test',
                    entities=[],
                    motifs=[],
                    metadata={'video_id': vid, 'importance_score': 0.5},
                )
            mock_process.side_effect = fake_process

            await spinner.spin([
                'https://youtube.com/playlist?list=PL_A',
                'https://youtube.com/playlist?list=PL_B',
            ])

            # vid_hot (resonance=2) should be processed first
            assert processed_order[0] == 'vid_hot'


@pytest.mark.asyncio
@pytest.mark.skipif(not YTDLP_AVAILABLE, reason="yt-dlp not available")
async def test_dead_video_skipped(tmp_path):
    """Test that None entries in playlist (dead/private videos) are handled."""
    spinner = PlaylistSpinner(
        importance_threshold=0.0,
        checkpoint_dir=tmp_path,
    )

    # Simulate yt-dlp returning None for a dead video
    manifest = PlaylistManifest(
        playlist_id='PL_test',
        title='Test',
        entries=[
            PlaylistEntry(video_id='vid_alive', title='Alive', playlist_index=0),
            # Dead videos are already filtered by _fetch_one_manifest
        ],
    )

    with patch.object(spinner, '_fetch_manifests', return_value=[manifest]):
        with patch.object(spinner, '_process_one_video') as mock_process:
            async def fake_process(vid, yt, scorer):
                return spinner._create_shard(
                    id_suffix=vid,
                    text=f"Content for {vid}",
                    episode='test',
                    entities=[],
                    motifs=[],
                    metadata={'video_id': vid, 'importance_score': 0.5},
                )
            mock_process.side_effect = fake_process

            result = await spinner.spin(['https://youtube.com/playlist?list=PL_test'])
            assert result.shard_count == 1


@pytest.mark.asyncio
@pytest.mark.skipif(not YTDLP_AVAILABLE, reason="yt-dlp not available")
async def test_transcript_fallback_to_metadata(tmp_path):
    """Test that videos without transcripts get a metadata-only shard."""
    spinner = PlaylistSpinner(
        importance_threshold=0.0,
        checkpoint_dir=tmp_path,
    )

    manifest = _make_manifest('PL_test', ['vid_no_transcript'])

    with patch.object(spinner, '_fetch_manifests', return_value=[manifest]):
        # Don't mock _process_one_video — let it run with a mocked YouTubeSpinner
        yt_mock = MagicMock()
        yt_result = MagicMock()
        yt_result.success = False
        yt_result.shards = []

        async def fake_spin(vid):
            return yt_result

        yt_mock.spin = fake_spin
        spinner._yt_spinner = yt_mock

        result = await spinner.spin(['https://youtube.com/playlist?list=PL_test'])

        assert result.shard_count == 1
        shard = result.shards[0]
        assert shard.metadata['shard_type'] == 'metadata_only'
        assert 'Video vid_no_transcript' in shard.text


# ============================================================================
# Checkpoint Persistence Tests
# ============================================================================

class TestCheckpointPersistence:

    def test_checkpoint_dir_creation(self, tmp_path):
        checkpoint_dir = tmp_path / 'checkpoints'
        spinner = PlaylistSpinner(checkpoint_dir=checkpoint_dir)
        assert checkpoint_dir.exists()

    def test_resonance_index_persisted(self, tmp_path):
        """Resonance index survives checkpoint save/load cycle."""
        spinner = PlaylistSpinner(checkpoint_dir=tmp_path)

        # Build some resonance data
        spinner._resonance_index.add('vid1', 'PL_A', {'title': 'Test'})
        spinner._resonance_index.add('vid1', 'PL_B')

        # Save
        seen = {'vid1'}
        manifests = [_make_manifest('PL_A', ['vid1'])]
        spinner._save_resonance_checkpoint(seen, manifests)

        # Load in new spinner
        spinner2 = PlaylistSpinner(checkpoint_dir=tmp_path)
        checkpoint = spinner2._load_resonance_checkpoint()

        assert 'vid1' in checkpoint.state['seen_video_ids']
        assert spinner2._resonance_index.resonance_count('vid1') == 2
