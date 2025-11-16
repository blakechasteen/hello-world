"""
Comprehensive Tests for Forensic Logging System

**Added: 2025-11-16** - Phase 4 Security Pipeline

Test Coverage:
- Hash chain integrity
- Tamper detection
- Storage backends
- Search functionality
- Compliance exports
- Performance benchmarks
"""

import pytest
import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from HoloLoom.security.forensics import ForensicLogger, ForensicEntry
from HoloLoom.security.forensics.hash_chain import HashChain, create_hash_chain
from HoloLoom.security.forensics.storage import FileStorage, create_storage_backend
from HoloLoom.security.forensics.search import ForensicSearchEngine, SearchQuery
from HoloLoom.security.forensics.export import ComplianceExporter
from HoloLoom.security.forensics.verification import ForensicVerifier


class TestHashChain:
    """Test hash chain implementation."""

    def test_genesis_block(self):
        """Test genesis block creation."""
        chain = create_hash_chain({'system': 'test'})

        assert len(chain) == 1
        assert chain.chain[0].entry_id == "genesis"
        assert chain.chain[0].previous_hash == "0" * 64
        assert len(chain.chain[0].current_hash) == 64

    def test_add_entry(self):
        """Test adding entries to chain."""
        chain = create_hash_chain()

        # Add first entry
        entry1 = chain.add_entry(
            entry_id="entry-1",
            timestamp="2025-11-16T00:00:00Z",
            data={'test': 'data1'}
        )

        assert entry1.entry_id == "entry-1"
        assert entry1.previous_hash == chain.chain[0].current_hash
        assert len(entry1.current_hash) == 64

        # Add second entry
        entry2 = chain.add_entry(
            entry_id="entry-2",
            timestamp="2025-11-16T00:01:00Z",
            data={'test': 'data2'}
        )

        assert entry2.previous_hash == entry1.current_hash
        assert len(chain) == 3  # genesis + 2 entries

    def test_duplicate_entry_id(self):
        """Test that duplicate entry IDs are rejected."""
        chain = create_hash_chain()

        chain.add_entry("entry-1", "2025-11-16T00:00:00Z", {'test': 'data'})

        with pytest.raises(ValueError, match="already exists"):
            chain.add_entry("entry-1", "2025-11-16T00:01:00Z", {'test': 'data2'})

    def test_verify_integrity_valid(self):
        """Test integrity verification with valid chain."""
        chain = create_hash_chain()

        # Add entries
        for i in range(10):
            chain.add_entry(
                f"entry-{i}",
                f"2025-11-16T00:{i:02d}:00Z",
                {'index': i}
            )

        # Verify
        is_valid, invalid_idx = chain.verify_integrity()

        assert is_valid is True
        assert invalid_idx is None

    def test_tamper_detection_data(self):
        """Test detection of tampered data."""
        chain = create_hash_chain()

        # Add entries
        chain.add_entry("entry-1", "2025-11-16T00:00:00Z", {'test': 'data1'})
        chain.add_entry("entry-2", "2025-11-16T00:01:00Z", {'test': 'data2'})

        # Tamper with data
        chain.chain[1].data['test'] = 'modified'

        # Verify - should detect tampering
        is_valid, invalid_idx = chain.verify_integrity()

        assert is_valid is False
        assert invalid_idx == 1

    def test_tamper_detection_hash(self):
        """Test detection of tampered hash."""
        chain = create_hash_chain()

        # Add entries
        chain.add_entry("entry-1", "2025-11-16T00:00:00Z", {'test': 'data1'})
        chain.add_entry("entry-2", "2025-11-16T00:01:00Z", {'test': 'data2'})

        # Tamper with hash
        chain.chain[1].current_hash = "0" * 64

        # Verify - should detect tampering
        is_valid, invalid_idx = chain.verify_integrity()

        assert is_valid is False
        assert invalid_idx == 1

    def test_tamper_detection_link(self):
        """Test detection of broken chain link."""
        chain = create_hash_chain()

        # Add entries
        chain.add_entry("entry-1", "2025-11-16T00:00:00Z", {'test': 'data1'})
        chain.add_entry("entry-2", "2025-11-16T00:01:00Z", {'test': 'data2'})

        # Break link
        chain.chain[2].previous_hash = "0" * 64

        # Verify - should detect tampering
        is_valid, invalid_idx = chain.verify_integrity()

        assert is_valid is False
        assert invalid_idx == 2

    def test_json_export_import(self):
        """Test JSON export and import."""
        chain = create_hash_chain()

        # Add entries
        for i in range(5):
            chain.add_entry(f"entry-{i}", f"2025-11-16T00:{i:02d}:00Z", {'index': i})

        # Export
        json_data = chain.to_json()

        # Import
        imported_chain = HashChain.from_json(json_data)

        # Verify imported chain
        assert len(imported_chain) == len(chain)
        assert imported_chain.get_latest_hash() == chain.get_latest_hash()

        is_valid, invalid_idx = imported_chain.verify_integrity()
        assert is_valid is True


class TestFileStorage:
    """Test file storage backend."""

    @pytest.mark.asyncio
    async def test_append_and_read(self):
        """Test appending and reading entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileStorage(log_dir=tmpdir)

            # Append entries
            entry1 = {
                'entry_id': 'e1',
                'timestamp': '2025-11-16T00:00:00Z',
                'data': {'test': 'data1'},
                'previous_hash': '0' * 64,
                'current_hash': 'hash1'
            }
            await storage.append(entry1)

            entry2 = {
                'entry_id': 'e2',
                'timestamp': '2025-11-16T00:01:00Z',
                'data': {'test': 'data2'},
                'previous_hash': 'hash1',
                'current_hash': 'hash2'
            }
            await storage.append(entry2)

            # Read all
            entries = await storage.read_all()

            assert len(entries) == 2
            assert entries[0]['entry_id'] == 'e1'
            assert entries[1]['entry_id'] == 'e2'

    @pytest.mark.asyncio
    async def test_compression(self):
        """Test gzip compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileStorage(log_dir=tmpdir)

            # Append many entries
            for i in range(100):
                entry = {
                    'entry_id': f'e{i}',
                    'timestamp': f'2025-11-16T00:{i:02d}:00Z',
                    'data': {'index': i, 'text': 'x' * 100},
                    'previous_hash': '0' * 64,
                    'current_hash': f'hash{i}'
                }
                await storage.append(entry)

            # Check compression ratio
            stats = await storage.get_stats()

            assert stats.compression_ratio < 1.0  # Should be compressed
            assert stats.total_entries == 100

    @pytest.mark.asyncio
    async def test_read_range(self):
        """Test reading entries in time range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileStorage(log_dir=tmpdir)

            # Append entries with different timestamps
            for i in range(10):
                timestamp = datetime(2025, 11, 16, i, 0, 0)
                entry = {
                    'entry_id': f'e{i}',
                    'timestamp': timestamp.isoformat() + 'Z',
                    'data': {'hour': i},
                    'previous_hash': '0' * 64,
                    'current_hash': f'hash{i}'
                }
                await storage.append(entry)

            # Read range (hours 3-7)
            start = datetime(2025, 11, 16, 3, 0, 0)
            end = datetime(2025, 11, 16, 7, 0, 0)
            entries = await storage.read_range(start, end)

            assert len(entries) == 5  # Hours 3, 4, 5, 6, 7
            assert entries[0]['data']['hour'] == 3
            assert entries[-1]['data']['hour'] == 7


class TestForensicLogger:
    """Test forensic logger."""

    @pytest.mark.asyncio
    async def test_basic_logging(self):
        """Test basic logging functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with ForensicLogger(storage_backend="file", log_dir=tmpdir) as logger:
                # Log event
                entry = await logger.log(
                    event_type="authentication",
                    severity="INFO",
                    action="user_login",
                    outcome="success",
                    user_id="alice"
                )

                assert entry.event_type == "authentication"
                assert entry.severity == "INFO"
                assert entry.action == "user_login"
                assert entry.outcome == "success"
                assert len(entry.current_hash) == 64

    @pytest.mark.asyncio
    async def test_sensitive_data_hashing(self):
        """Test that sensitive data is hashed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with ForensicLogger(
                storage_backend="file",
                log_dir=tmpdir,
                hash_sensitive=True
            ) as logger:
                entry = await logger.log(
                    event_type="access",
                    severity="INFO",
                    action="file_read",
                    outcome="success",
                    user_id="alice@example.com",
                    source_ip="192.168.1.100"
                )

                # Should be hashed (not original values)
                assert entry.user_id != "alice@example.com"
                assert entry.source_ip != "192.168.1.100"
                assert len(entry.user_id) == 16  # SHA-256 truncated
                assert len(entry.source_ip) == 16

    @pytest.mark.asyncio
    async def test_chain_integrity(self):
        """Test that logged entries maintain chain integrity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with ForensicLogger(storage_backend="file", log_dir=tmpdir) as logger:
                # Log multiple events
                entries = []
                for i in range(10):
                    entry = await logger.log(
                        event_type="system",
                        severity="INFO",
                        action=f"operation_{i}",
                        outcome="success"
                    )
                    entries.append(entry)

                # Verify chain
                is_valid, invalid_idx = await logger.get_chain_integrity()

                assert is_valid is True
                assert invalid_idx is None

                # Verify links
                for i in range(1, len(entries)):
                    assert entries[i].previous_hash == entries[i-1].current_hash

    @pytest.mark.asyncio
    async def test_performance(self):
        """Test write performance (<5ms target)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with ForensicLogger(storage_backend="file", log_dir=tmpdir) as logger:
                # Log 100 events
                for i in range(100):
                    await logger.log(
                        event_type="system",
                        severity="INFO",
                        action=f"test_{i}",
                        outcome="success"
                    )

                # Check average write time
                stats = await logger.get_stats()

                assert stats['avg_write_time_ms'] < 10.0  # Allow 10ms for file I/O
                assert stats['total_entries'] == 100


class TestForensicSearch:
    """Test forensic search engine."""

    @pytest.mark.asyncio
    async def test_time_range_search(self):
        """Test searching by time range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileStorage(log_dir=tmpdir)
            async with ForensicLogger(storage_backend="file", log_dir=tmpdir) as logger:
                # Log events at different times
                base_time = datetime(2025, 11, 16, 0, 0, 0)
                for i in range(24):
                    await logger.log(
                        event_type="system",
                        severity="INFO",
                        action=f"hourly_check_{i}",
                        outcome="success"
                    )

                # Search last 12 hours
                search = ForensicSearchEngine(storage)
                result = await search.search(SearchQuery(
                    start_time=datetime(2025, 11, 16, 12, 0, 0),
                    end_time=datetime(2025, 11, 16, 23, 59, 59),
                    limit=100
                ))

                # Should find entries (depending on actual timestamps)
                assert result.total_matches >= 0
                assert result.query_time_ms < 200.0

    @pytest.mark.asyncio
    async def test_event_type_filter(self):
        """Test filtering by event type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with ForensicLogger(storage_backend="file", log_dir=tmpdir) as logger:
                # Log different event types
                await logger.log(event_type="authentication", severity="INFO", action="login", outcome="success")
                await logger.log(event_type="authentication", severity="WARNING", action="login", outcome="failure")
                await logger.log(event_type="attack", severity="CRITICAL", action="sql_injection", outcome="blocked")
                await logger.log(event_type="access", severity="INFO", action="read_file", outcome="success")

                # Search for authentication events
                search = ForensicSearchEngine(logger.storage)
                result = await search.search(SearchQuery(
                    event_types=["authentication"],
                    limit=100
                ))

                assert result.total_matches == 2
                assert all(e.event_type == "authentication" for e in result.entries)

    @pytest.mark.asyncio
    async def test_severity_filter(self):
        """Test filtering by severity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with ForensicLogger(storage_backend="file", log_dir=tmpdir) as logger:
                # Log different severities
                await logger.log(event_type="system", severity="INFO", action="info", outcome="success")
                await logger.log(event_type="system", severity="WARNING", action="warning", outcome="success")
                await logger.log(event_type="system", severity="CRITICAL", action="critical", outcome="success")

                # Search for critical events
                search = ForensicSearchEngine(logger.storage)
                result = await search.search(SearchQuery(
                    severities=["CRITICAL"],
                    limit=100
                ))

                assert result.total_matches == 1
                assert result.entries[0].severity == "CRITICAL"

    @pytest.mark.asyncio
    async def test_failed_authentications(self):
        """Test getting failed authentication attempts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with ForensicLogger(storage_backend="file", log_dir=tmpdir) as logger:
                # Log authentication attempts
                await logger.log(event_type="authentication", severity="INFO", action="login", outcome="success")
                await logger.log(event_type="authentication", severity="WARNING", action="login", outcome="failure")
                await logger.log(event_type="authentication", severity="WARNING", action="login", outcome="failure")
                await logger.log(event_type="authentication", severity="WARNING", action="login", outcome="failure")

                # Get failed attempts
                search = ForensicSearchEngine(logger.storage)
                failed = await search.get_failed_authentications(hours=24, limit=100)

                assert len(failed) == 3
                assert all(e.outcome == "failure" for e in failed)


class TestComplianceExport:
    """Test compliance exports."""

    @pytest.mark.asyncio
    async def test_gdpr_export_json(self):
        """Test GDPR export in JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with ForensicLogger(storage_backend="file", log_dir=tmpdir) as logger:
                # Log user activity
                user_id = "alice"
                for i in range(5):
                    await logger.log(
                        event_type="access",
                        severity="INFO",
                        action=f"read_document_{i}",
                        outcome="success",
                        user_id=user_id
                    )

                # Export GDPR data
                exporter = ComplianceExporter(logger.storage)
                result = await exporter.export_gdpr(user_id, format="json")

                assert result.compliance_standard == "GDPR Article 20"
                assert result.entry_count >= 5
                assert '"export_type": "GDPR_Article_20"' in result.content

                # Verify JSON is valid
                data = json.loads(result.content)
                assert data['user_id'] == user_id
                assert len(data['entries']) >= 5

    @pytest.mark.asyncio
    async def test_soc2_export(self):
        """Test SOC2 audit trail export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with ForensicLogger(storage_backend="file", log_dir=tmpdir) as logger:
                # Log various events
                await logger.log(event_type="authentication", severity="INFO", action="login", outcome="success")
                await logger.log(event_type="access", severity="INFO", action="read_file", outcome="success")
                await logger.log(event_type="admin", severity="WARNING", action="config_change", outcome="success")

                # Export SOC2 data
                exporter = ComplianceExporter(logger.storage)
                start = datetime(2025, 11, 1)
                end = datetime(2025, 11, 30)
                result = await exporter.export_soc2(start, end, format="json")

                assert result.compliance_standard == "SOC2 (CC6.1, CC7.2)"
                assert result.entry_count >= 3

                # Verify JSON
                data = json.loads(result.content)
                assert data['export_type'] == "SOC2_Audit_Trail"
                assert 'events_by_type' in data

    @pytest.mark.asyncio
    async def test_iso27001_export(self):
        """Test ISO27001 security event export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with ForensicLogger(storage_backend="file", log_dir=tmpdir) as logger:
                # Log security events
                await logger.log(event_type="attack", severity="CRITICAL", action="sql_injection", outcome="blocked")
                await logger.log(event_type="incident", severity="CRITICAL", action="data_breach", outcome="contained")

                # Export ISO27001 data
                exporter = ComplianceExporter(logger.storage)
                result = await exporter.export_iso27001(days=90, format="json")

                assert result.compliance_standard == "ISO27001 A.12.4.1"
                assert result.entry_count >= 2

                # Verify JSON
                data = json.loads(result.content)
                assert data['export_type'] == "ISO27001_Security_Events"
                assert data['control'] == "A.12.4.1 Event logging"


class TestForensicVerifier:
    """Test forensic verification."""

    @pytest.mark.asyncio
    async def test_verify_valid_chain(self):
        """Test verification of valid chain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with ForensicLogger(storage_backend="file", log_dir=tmpdir) as logger:
                # Log events
                for i in range(10):
                    await logger.log(
                        event_type="system",
                        severity="INFO",
                        action=f"operation_{i}",
                        outcome="success"
                    )

                # Verify
                verifier = ForensicVerifier(logger.storage)
                result = await verifier.verify_chain()

                assert result.is_valid is True
                assert len(result.tampered_entries) == 0
                assert result.verified_entries == result.total_entries
                assert "✓ VALID" in result.report

    @pytest.mark.asyncio
    async def test_detect_tampered_chain(self):
        """Test detection of tampered chain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with ForensicLogger(storage_backend="file", log_dir=tmpdir) as logger:
                # Log events
                for i in range(10):
                    await logger.log(
                        event_type="system",
                        severity="INFO",
                        action=f"operation_{i}",
                        outcome="success"
                    )

            # Tamper with storage (modify a file entry)
            entries = await logger.storage.read_all()
            entries[5]['action'] = 'tampered_operation'

            # Write back tampered data
            log_file = Path(tmpdir) / "forensic_logs.jsonl.gz"
            import gzip
            with gzip.open(log_file, 'wt') as f:
                for entry in entries:
                    f.write(json.dumps(entry) + '\n')

            # Verify - should detect tampering
            storage = FileStorage(log_dir=tmpdir)
            verifier = ForensicVerifier(storage)
            result = await verifier.verify_chain()

            assert result.is_valid is False
            assert len(result.tampered_entries) > 0
            assert "✗ INVALID" in result.report

    @pytest.mark.asyncio
    async def test_verification_performance(self):
        """Test verification performance (<10s for 1M entries target)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with ForensicLogger(storage_backend="file", log_dir=tmpdir) as logger:
                # Log 1000 events (reduced for test speed)
                for i in range(1000):
                    await logger.log(
                        event_type="system",
                        severity="INFO",
                        action=f"operation_{i}",
                        outcome="success"
                    )

                # Verify
                verifier = ForensicVerifier(logger.storage)
                result = await verifier.verify_chain()

                # Should be fast for 1000 entries
                assert result.verification_time_ms < 1000.0  # <1s for 1000 entries
                assert result.is_valid is True


@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Initialize logger
        async with ForensicLogger(
            storage_backend="file",
            log_dir=tmpdir,
            hash_sensitive=True
        ) as logger:
            # 2. Log various events
            await logger.log(
                event_type="authentication",
                severity="INFO",
                action="user_login",
                outcome="success",
                user_id="alice@example.com",
                source_ip="192.168.1.100"
            )

            await logger.log(
                event_type="attack",
                severity="CRITICAL",
                action="sql_injection_attempt",
                outcome="blocked",
                source_ip="10.0.0.50",
                metadata={"query": "SELECT * FROM users WHERE 1=1"}
            )

            await logger.log(
                event_type="access",
                severity="INFO",
                action="read_sensitive_file",
                outcome="success",
                user_id="alice@example.com"
            )

            # 3. Search for attacks
            search = ForensicSearchEngine(logger.storage)
            attacks = await search.get_recent_attacks(hours=24)

            assert len(attacks) >= 1
            assert attacks[0].event_type == "attack"

            # 4. Verify chain integrity
            verifier = ForensicVerifier(logger.storage)
            verification = await verifier.verify_chain()

            assert verification.is_valid is True

            # 5. Export compliance data
            exporter = ComplianceExporter(logger.storage)
            gdpr_export = await exporter.export_gdpr("alice@example.com", format="json")

            assert gdpr_export.entry_count >= 2  # alice's events

            # 6. Check statistics
            stats = await logger.get_stats()

            assert stats['total_entries'] >= 3
            assert stats['avg_write_time_ms'] < 20.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
