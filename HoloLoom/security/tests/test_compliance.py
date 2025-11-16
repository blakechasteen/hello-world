"""
Compliance Framework Tests

**Created: 2025-11-16**
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from HoloLoom.security.compliance import (
    ComplianceMonitor,
    ComplianceFramework,
    ControlStatus,
    RiskLevel,
    SOC2Monitor,
    TrustServiceCriteria,
    GDPRMonitor,
    DataSubjectRight,
    ISO27001Monitor,
    EvidenceCollector,
    EvidenceType,
    ComplianceReporter,
    generate_compliance_report,
    generate_audit_package,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp = Path(tempfile.mkdtemp())
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def compliance_monitor(temp_dir):
    """Create compliance monitor"""
    return ComplianceMonitor(
        frameworks=[ComplianceFramework.SOC2, ComplianceFramework.GDPR],
        evidence_dir=temp_dir / "evidence",
        auto_collect=False
    )


@pytest.fixture
def soc2_monitor(temp_dir):
    """Create SOC2 monitor"""
    return SOC2Monitor(evidence_dir=temp_dir / "soc2")


@pytest.fixture
def gdpr_monitor():
    """Create GDPR monitor"""
    return GDPRMonitor()


@pytest.fixture
def iso27001_monitor():
    """Create ISO 27001 monitor"""
    return ISO27001Monitor()


@pytest.fixture
def evidence_collector(temp_dir):
    """Create evidence collector"""
    from HoloLoom.security.compliance.evidence import EvidenceRepository
    repo = EvidenceRepository(repo_path=temp_dir / "evidence_repo")
    return EvidenceCollector(repository=repo)


class TestComplianceMonitor:
    """Test core compliance monitor"""

    @pytest.mark.asyncio
    async def test_initialization(self, compliance_monitor):
        """Test monitor initialization"""
        assert len(compliance_monitor.frameworks) == 2
        assert ComplianceFramework.SOC2 in compliance_monitor.frameworks
        assert ComplianceFramework.GDPR in compliance_monitor.frameworks

    @pytest.mark.asyncio
    async def test_automated_checks(self, compliance_monitor):
        """Test automated compliance checks"""
        await compliance_monitor._run_automated_checks()

        # Check that automated checks ran
        assert compliance_monitor.last_monitoring is not None

    @pytest.mark.asyncio
    async def test_check_compliance(self, compliance_monitor):
        """Test compliance status check"""
        status = await compliance_monitor.check_compliance(ComplianceFramework.SOC2)

        assert status.framework == ComplianceFramework.SOC2
        assert 0.0 <= status.overall_score <= 1.0
        assert status.controls_total >= 0

    @pytest.mark.asyncio
    async def test_gap_identification(self, compliance_monitor):
        """Test compliance gap identification"""
        # Add a gap
        compliance_monitor._add_gap(
            control_id="TEST.1",
            framework=ComplianceFramework.SOC2,
            severity=RiskLevel.HIGH,
            description="Test gap",
            remediation="Fix it"
        )

        gaps = compliance_monitor.get_gaps_by_severity(RiskLevel.HIGH)
        assert len(gaps) >= 1

    @pytest.mark.asyncio
    async def test_export_import(self, compliance_monitor, temp_dir):
        """Test export and import of compliance status"""
        export_path = temp_dir / "compliance_export.json"

        # Export
        compliance_monitor.export_status(export_path)
        assert export_path.exists()

        # Import
        new_monitor = ComplianceMonitor()
        new_monitor.import_status(export_path)

        # Verify
        assert len(new_monitor.frameworks) > 0


class TestSOC2Monitor:
    """Test SOC2 monitor"""

    def test_initialization(self, soc2_monitor):
        """Test SOC2 monitor initialization"""
        assert len(soc2_monitor.controls) > 0
        assert "CC6.1" in soc2_monitor.controls

    def test_control_effectiveness(self, soc2_monitor):
        """Test control effectiveness check"""
        # Mark control as effective
        control = soc2_monitor.controls["CC6.1"]
        control.implemented = True
        control.tested = True
        control.effective = True

        assert soc2_monitor.check_control_effectiveness("CC6.1") is True

    @pytest.mark.asyncio
    async def test_evidence_collection(self, soc2_monitor):
        """Test automated evidence collection"""
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()

        evidence = await soc2_monitor.collect_evidence(start_date, end_date)

        # Should have collected evidence for automated controls
        assert len(evidence) > 0

    def test_readiness_report(self, soc2_monitor):
        """Test SOC2 readiness report generation"""
        report = soc2_monitor.generate_readiness_report()

        assert 'readiness_score' in report
        assert 'audit_ready' in report
        assert 'controls' in report
        assert 0.0 <= report['readiness_score'] <= 1.0


class TestGDPRMonitor:
    """Test GDPR monitor"""

    def test_initialization(self, gdpr_monitor):
        """Test GDPR monitor initialization"""
        assert len(gdpr_monitor.processing_activities) > 0

    def test_dpia_conduct(self, gdpr_monitor):
        """Test DPIA assessment"""
        dpia = gdpr_monitor.conduct_dpia(
            processing_description="Test processing",
            purposes=["Service improvement"],
            risks=["Re-identification"],
            assessor="Test Team"
        )

        assert dpia.assessment_id.startswith("DPIA_")
        assert len(dpia.purposes) > 0
        assert len(dpia.risks_identified) > 0
        assert dpia.conclusion in ["acceptable", "needs_consultation", "not_acceptable"]

    def test_data_subject_request(self, gdpr_monitor):
        """Test data subject request handling"""
        request = gdpr_monitor.receive_request(
            request_type=DataSubjectRight.ACCESS,
            email="test@example.com"
        )

        assert request.request_id.startswith("DSR_")
        assert request.status == "pending"
        assert request.due_date > datetime.now()

    def test_access_request_processing(self, gdpr_monitor):
        """Test access request processing"""
        request = gdpr_monitor.receive_request(
            request_type=DataSubjectRight.ACCESS,
            email="test@example.com"
        )

        data_package = gdpr_monitor.process_access_request(request)

        assert 'personal_data' in data_package
        assert 'rights_information' in data_package
        assert request.status == "completed"

    def test_erasure_request_processing(self, gdpr_monitor):
        """Test erasure request processing"""
        request = gdpr_monitor.receive_request(
            request_type=DataSubjectRight.ERASURE,
            email="test@example.com"
        )

        result = gdpr_monitor.process_erasure_request(request)

        assert result is True
        assert request.status == "completed"

    def test_compliance_check(self, gdpr_monitor):
        """Test GDPR compliance check"""
        status = gdpr_monitor.check_compliance()

        assert 'compliance_score' in status
        assert 'checks' in status
        assert 0.0 <= status['compliance_score'] <= 1.0


class TestISO27001Monitor:
    """Test ISO 27001 monitor"""

    def test_initialization(self, iso27001_monitor):
        """Test ISO 27001 monitor initialization"""
        assert len(iso27001_monitor.controls) > 0
        assert "A.9.1" in iso27001_monitor.controls

    def test_control_check(self, iso27001_monitor):
        """Test control check"""
        control = iso27001_monitor.check_control("A.9.1")

        assert control is not None
        assert control.control_id == "A.9.1"

    def test_risk_assessment(self, iso27001_monitor):
        """Test risk assessment"""
        risks = iso27001_monitor.conduct_risk_assessment()

        assert len(risks) > 0
        assert iso27001_monitor.last_risk_assessment is not None

        # Check risk structure
        for risk in risks:
            assert 'risk_id' in risk
            assert 'threat' in risk
            assert 'impact' in risk
            assert 'likelihood' in risk

    def test_isms_status(self, iso27001_monitor):
        """Test ISMS status"""
        status = iso27001_monitor.get_isms_status()

        assert status.scope == iso27001_monitor.scope
        assert status.controls_total > 0
        assert status.certification_status in ["not_certified", "in_progress", "certified"]

    def test_statement_of_applicability(self, iso27001_monitor):
        """Test SoA generation"""
        soa = iso27001_monitor.generate_statement_of_applicability()

        assert 'organization' in soa
        assert 'scope' in soa
        assert 'controls' in soa
        assert len(soa['controls']) > 0


class TestEvidenceCollector:
    """Test evidence collector"""

    @pytest.mark.asyncio
    async def test_access_log_collection(self, evidence_collector):
        """Test access log collection"""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        evidence = await evidence_collector.collect_access_logs(start_date, end_date)

        assert len(evidence) > 0
        assert evidence[0].evidence_type == EvidenceType.ACCESS_LOG

    @pytest.mark.asyncio
    async def test_change_log_collection(self, evidence_collector):
        """Test change log collection"""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        evidence = await evidence_collector.collect_change_logs(start_date, end_date)

        assert len(evidence) > 0
        assert evidence[0].evidence_type == EvidenceType.CHANGE_LOG

    @pytest.mark.asyncio
    async def test_training_records_collection(self, evidence_collector):
        """Test training records collection"""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        evidence = await evidence_collector.collect_training_records(start_date, end_date)

        assert len(evidence) > 0
        assert evidence[0].evidence_type == EvidenceType.TRAINING_RECORD

    @pytest.mark.asyncio
    async def test_audit_package_collection(self, evidence_collector):
        """Test complete audit package collection"""
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()

        package = await evidence_collector.collect_audit_package(start_date, end_date)

        assert 'access_logs' in package
        assert 'change_logs' in package
        assert 'training_records' in package

    def test_evidence_search(self, evidence_collector):
        """Test evidence search"""
        # Add some test evidence
        from HoloLoom.security.compliance.evidence import AuditEvidence

        evidence1 = AuditEvidence(
            evidence_id="TEST_001",
            evidence_type=EvidenceType.ACCESS_LOG,
            control_id="CC6.1",
            description="Test evidence"
        )

        evidence_collector.repository.add_evidence(evidence1)

        # Search
        results = evidence_collector.repository.search_evidence(
            control_id="CC6.1"
        )

        assert len(results) >= 1


class TestComplianceReporter:
    """Test compliance reporter"""

    @pytest.fixture
    def reporter(self, temp_dir):
        """Create compliance reporter"""
        return ComplianceReporter(output_dir=temp_dir / "reports")

    @pytest.mark.asyncio
    async def test_monthly_report(self, reporter):
        """Test monthly report generation"""
        report = await reporter.generate_monthly_report()

        assert 'report_type' in report
        assert report['report_type'] == "monthly_summary"
        assert 'soc2' in report
        assert 'gdpr' in report
        assert 'iso27001' in report

    def test_board_report(self, reporter):
        """Test board report generation"""
        report = reporter.generate_board_report()

        assert 'report_type' in report
        assert report['report_type'] == "quarterly_board_report"
        assert 'executive_summary' in report
        assert 'compliance_frameworks' in report

    @pytest.mark.asyncio
    async def test_audit_package(self, reporter):
        """Test audit package generation"""
        package = await reporter.generate_audit_package(
            framework=ComplianceFramework.SOC2
        )

        assert 'framework' in package
        assert package['framework'] == 'soc2'
        assert 'audit_period' in package
        assert 'evidence' in package

    def test_compliance_score(self, reporter):
        """Test compliance score calculation"""
        score = reporter.calculate_compliance_score(ComplianceFramework.SOC2)

        assert score.framework == ComplianceFramework.SOC2
        assert 0.0 <= score.score <= 1.0
        assert score.grade in ["A+", "A", "B", "C", "D", "F"]


class TestPublicAPI:
    """Test public API functions"""

    @pytest.mark.asyncio
    async def test_generate_compliance_report(self):
        """Test generate_compliance_report function"""
        report = await generate_compliance_report(
            frameworks=['soc2', 'gdpr'],
            output_format='json'
        )

        assert 'report_type' in report
        assert 'soc2' in report

    @pytest.mark.asyncio
    async def test_generate_audit_package(self):
        """Test generate_audit_package function"""
        package = await generate_audit_package(
            framework='soc2',
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now()
        )

        assert 'framework' in package
        assert package['framework'] == 'soc2'


# Run tests with: pytest HoloLoom/security/tests/test_compliance.py -v
