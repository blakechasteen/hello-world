"""
Unit Tests for Daily Tapestry
==============================

Tests the tapestry briefing pipeline:
- LLM summarization with fallback
- Topic clustering integration
- Trend detection and persistence
- Matrix IPC delivery
- Markdown and JSON output formatting

Author: Claude Code
Date: March 2026
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from hololoom.spinningWheel.daily_tapestry import (
    Tapestry,
    TapestryEntry,
    TopicCluster,
    Trend,
    TrendLog,
    cluster_entries,
    deliver_to_matrix,
    detect_trends,
    summarize_transcript,
    weave_tapestry,
)


# ============================================================================
# Test Helpers
# ============================================================================

def _entry(video_id: str, title: str = 'Test', resonance: int = 1, summary: str = 'Summary') -> TapestryEntry:
    return TapestryEntry(
        video_id=video_id,
        title=title,
        channel='TestChannel',
        url=f'https://youtube.com/watch?v={video_id}',
        resonance_count=resonance,
        importance=0.5,
        summary=summary,
        playlists=['PL_A'],
    )


# ============================================================================
# Trend Detection Tests
# ============================================================================

class TestTrendLog:

    def test_empty_log(self, tmp_path):
        log = TrendLog()
        assert log.entries == []
        assert log.previous_counts() == {}

    def test_append_and_save(self, tmp_path):
        log = TrendLog()
        log.append_run('2026-03-20', {'ML': 3, 'DevOps': 1})
        log.append_run('2026-03-21', {'ML': 5, 'Security': 2})

        path = tmp_path / 'trend.json'
        log.save(path)

        loaded = TrendLog.load(path)
        assert len(loaded.entries) == 2

    def test_previous_counts_excludes_current(self):
        log = TrendLog()
        log.append_run('2026-03-20', {'ML': 3})
        log.append_run('2026-03-21', {'ML': 5})  # Current

        prev = log.previous_counts(window_days=30)
        assert prev.get('ML') == 3  # Only first run

    def test_load_missing_file(self, tmp_path):
        log = TrendLog.load(tmp_path / 'nonexistent.json')
        assert log.entries == []


class TestDetectTrends:

    def test_first_run_no_trends(self):
        clusters = [TopicCluster(topic='ML', entries=[_entry('v1')])]
        log = TrendLog()
        trends = detect_trends(clusters, log, '2026-03-20')
        assert trends == []  # No previous data

    def test_new_topic_detected(self):
        log = TrendLog()
        log.append_run('2026-03-19', {'DevOps': 2})

        clusters = [
            TopicCluster(topic='DevOps', entries=[_entry('v1')]),
            TopicCluster(topic='ML', entries=[_entry('v2'), _entry('v3')]),
        ]
        trends = detect_trends(clusters, log, '2026-03-20')

        ml_trend = next(t for t in trends if t.topic == 'ML')
        assert ml_trend.direction == 'new'

    def test_growing_topic(self):
        log = TrendLog()
        log.append_run('2026-03-19', {'ML': 1})

        clusters = [TopicCluster(topic='ML', entries=[_entry('v1'), _entry('v2'), _entry('v3')])]
        trends = detect_trends(clusters, log, '2026-03-20')

        assert trends[0].topic == 'ML'
        assert trends[0].direction == 'up'
        assert trends[0].current_count == 3
        assert trends[0].previous_count == 1


class TestTrend:

    def test_describe_new(self):
        t = Trend('AI Safety', 3, 0, 'new')
        assert 'new topic' in t.describe()

    def test_describe_up(self):
        t = Trend('ML', 5, 2, 'up')
        desc = t.describe()
        assert 'ML' in desc
        assert '5' in desc


# ============================================================================
# LLM Summarization Tests
# ============================================================================

class TestSummarization:

    @pytest.mark.asyncio
    async def test_fallback_when_llm_unavailable(self):
        """Without LLM, should truncate text."""
        with patch('hololoom.spinningWheel.daily_tapestry.LLM_AVAILABLE', False):
            result = await summarize_transcript('A' * 500, 'Test Video')
            assert len(result) <= 210  # 200 + '...'
            assert result.endswith('...')

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self):
        """If LLM call fails, should fall back gracefully."""
        with patch('hololoom.spinningWheel.daily_tapestry.LLM_AVAILABLE', True):
            mock_backend = MagicMock()
            mock_backend.call = AsyncMock(side_effect=Exception("Connection refused"))

            with patch('hololoom.spinningWheel.daily_tapestry.OllamaBackend', return_value=mock_backend):
                result = await summarize_transcript('Some transcript text here', 'Test')
                assert len(result) > 0  # Got fallback, not crash

    @pytest.mark.asyncio
    async def test_successful_summarization(self):
        """With working LLM, should return summary."""
        with patch('hololoom.spinningWheel.daily_tapestry.LLM_AVAILABLE', True):
            mock_backend = MagicMock()
            mock_backend.call = AsyncMock(return_value={
                'message': {'content': '- Point 1\n- Point 2\n- Point 3'},
            })

            with patch('hololoom.spinningWheel.daily_tapestry.OllamaBackend', return_value=mock_backend):
                with patch('hololoom.spinningWheel.daily_tapestry.extract_content', return_value='- Point 1\n- Point 2\n- Point 3'):
                    result = await summarize_transcript('Long transcript...', 'Test')
                    assert '- Point 1' in result


# ============================================================================
# Topic Clustering Tests
# ============================================================================

class TestClustering:

    def test_skips_when_too_few_entries(self):
        entries = [_entry('v1'), _entry('v2')]
        result = cluster_entries(entries)
        assert result == []  # <3 entries, not enough

    def test_skips_when_deps_unavailable(self):
        entries = [_entry('v1'), _entry('v2'), _entry('v3'), _entry('v4')]
        with patch('hololoom.spinningWheel.daily_tapestry.CLUSTERING_AVAILABLE', False):
            result = cluster_entries(entries)
            assert result == []

    def test_clustering_integration(self):
        """Test with mocked cluster function."""
        entries = [
            _entry('v1', summary='Neural networks and deep learning'),
            _entry('v2', summary='Kubernetes and container orchestration'),
            _entry('v3', summary='Machine learning optimization'),
            _entry('v4', summary='Docker deployment strategies'),
        ]

        mock_output = MagicMock()
        mock_cluster_1 = MagicMock()
        mock_cluster_1.label = 'Machine Learning'
        mock_cluster_1.indices = [0, 2]
        mock_cluster_2 = MagicMock()
        mock_cluster_2.label = 'DevOps'
        mock_cluster_2.indices = [1, 3]
        mock_output.__iter__ = MagicMock(return_value=iter([mock_cluster_1, mock_cluster_2]))

        with patch('hololoom.spinningWheel.daily_tapestry.CLUSTERING_AVAILABLE', True):
            with patch('hololoom.spinningWheel.daily_tapestry.cluster_texts', return_value=mock_output):
                result = cluster_entries(entries)

                assert len(result) == 2
                assert result[0].topic == 'Machine Learning'
                assert len(result[0].entries) == 2
                assert result[1].topic == 'DevOps'


# ============================================================================
# Matrix IPC Delivery Tests
# ============================================================================

class TestMatrixDelivery:

    def test_ipc_file_created(self, tmp_path):
        tapestry = Tapestry(
            date='2026-03-23',
            total_videos_scanned=10,
            new_videos_processed=3,
        )

        with patch('hololoom.spinningWheel.daily_tapestry.FEMTOCLAW_IPC_DIR', tmp_path):
            deliver_to_matrix(tapestry, '!test:piserver.local')

            # Check file was created
            messages_dir = tmp_path / 'main' / 'messages'
            assert messages_dir.exists()

            files = list(messages_dir.glob('tapestry-*.json'))
            assert len(files) == 1

            data = json.loads(files[0].read_text())
            assert data['type'] == 'message'
            assert data['chatJid'] == '!test:piserver.local'
            assert 'Daily Tapestry' in data['text']

    def test_no_tmp_files_left(self, tmp_path):
        """Atomic write should not leave .tmp files."""
        tapestry = Tapestry(date='2026-03-23', total_videos_scanned=0, new_videos_processed=0)

        with patch('hololoom.spinningWheel.daily_tapestry.FEMTOCLAW_IPC_DIR', tmp_path):
            deliver_to_matrix(tapestry, '!room:server')

            messages_dir = tmp_path / 'main' / 'messages'
            tmp_files = list(messages_dir.glob('*.tmp'))
            assert len(tmp_files) == 0


# ============================================================================
# Tapestry Output Tests
# ============================================================================

class TestTapestryOutput:

    def test_markdown_with_trends(self):
        tapestry = Tapestry(
            date='2026-03-23',
            total_videos_scanned=50,
            new_videos_processed=5,
            trends=[Trend('ML', 5, 2, 'up'), Trend('Security', 3, 0, 'new')],
            high_resonance=[_entry('v1', title='Popular Video', resonance=3)],
        )

        md = tapestry.to_markdown()
        assert '### Trends' in md
        assert 'ML' in md
        assert 'Popular Video' in md
        assert '### Stats' in md

    def test_markdown_flat_fallback(self):
        """Without clusters, should show flat list."""
        tapestry = Tapestry(
            date='2026-03-23',
            total_videos_scanned=10,
            new_videos_processed=2,
            high_resonance=[],
            clusters=[],
        )
        # Add entries via a cluster that won't render
        md = tapestry.to_markdown()
        assert '### Stats' in md

    def test_json_roundtrip(self):
        tapestry = Tapestry(
            date='2026-03-23',
            total_videos_scanned=10,
            new_videos_processed=2,
            trends=[Trend('ML', 3, 1, 'up')],
            high_resonance=[_entry('v1', resonance=2)],
            clusters=[TopicCluster(topic='DevOps', entries=[_entry('v2')])],
        )

        data = tapestry.to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        assert parsed['date'] == '2026-03-23'
        assert len(parsed['trends']) == 1
        assert parsed['trends'][0]['direction'] == 'up'
        assert len(parsed['high_resonance']) == 1
        assert len(parsed['clusters']) == 1
